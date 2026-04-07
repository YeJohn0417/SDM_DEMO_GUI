import sys
import os
import numpy as np
import datetime
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QLabel, QPushButton, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg
from scipy import signal

# 如果在 Windows 或 Mac 執行請註解掉這行；樹莓派請保留
#os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    import filter           
    from spi_receive import SerialStreaming
except ImportError:
    print(">>> [警告] 找不到 filter.py 或 spi_receive.py")

class SPIWorker(QThread):
    # 🌟 訊號通道：傳遞 (原始未處理的 bits, 濾波處理過的 final_ecg)
    data_ready = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        self.stream = SerialStreaming() 
        if not self.stream.connect(): return
        
        comp = np.genfromtxt('coef2.dat').flatten()
        fir = np.genfromtxt('coef_fir_v2.dat').flatten()
        
        fs_hw = 24000 
        sos_band = signal.butter(3, [1.0, 35], fs=fs_hw, btype='band', output='sos')
        zi_band = signal.sosfilt_zi(sos_band)

        while self._is_running:
            bits = self.stream.get_chunk()
            if bits is not None:
                try:
                    # 給畫面用的處理資料
                    fir_out, _ = filter.filter(bits, comp, fir)
                    final_ecg, zi_band = signal.sosfilt(sos_band, fir_out, zi=zi_band)
                    
                    # 🌟 同時把「原始 bits」和「處理後的 final_ecg」送給 GUI
                    self.data_ready.emit(bits, final_ecg)
                except: pass
            else:
                self.msleep(1)

    def stop(self):
        self._is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Dataset Collector (Display Filtered -> Save RAW)")
        self.resize(1100, 800)
        self.setStyleSheet("background-color: #000;")

        # --- 參數設定 ---
        self.fs_plot = 24000           # 濾波後的採樣率
        self.current_window_sec = 2    # 畫面顯示與擷取的秒數
        
        # 1. 顯示用的 Buffer (固定大小)
        self.plot_buffer_size = self.fs_plot * self.current_window_sec
        self.plot_buffer = np.zeros(self.plot_buffer_size)
        
        # 2. 存檔用的 RAW Deque (動態大小，收到第一筆資料時初始化)
        self.raw_deque = None 
        
        # 建立 dataset 存放資料夾
        self.dataset_dir = "./dataset"
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.setup_ui()
        
        # X 軸時間陣列 (為了顯示效能，UI 上每 4 個點畫 1 個點)
        self.ds_factor = 4 
        self.plot_time = np.linspace(-self.current_window_sec, 0, self.plot_buffer_size // self.ds_factor)

    def setup_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        
        # --- 狀態列 ---
        info_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #00FF41; font-family: Monospace; font-size: 28px; font-weight: bold;")
        info_layout.addWidget(self.status_label)
        
        self.mode_label = QLabel("Visual: Filtered | Save: RAW (2s)")
        self.mode_label.setStyleSheet("color: #00FFFF; font-family: Monospace; font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.mode_label, alignment=Qt.AlignRight)
        layout.addLayout(info_layout)

        # --- 繪圖區 ---
        self.pw = pg.PlotWidget()
        self.pw.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.pw.plot(pen=pg.mkPen(color='#00FFFF', width=2))
        layout.addWidget(self.pw)

        # --- 按鈕區 ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("START MONITORING")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold; font-size: 16px; border-radius: 10px;")
        self.start_btn.clicked.connect(self.toggle_detection)
        btn_layout.addWidget(self.start_btn)

        self.capture_btn = QPushButton("📸 CAPTURE 2s RAW DATA")
        self.capture_btn.setMinimumHeight(60)
        self.capture_btn.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold; font-size: 16px; border-radius: 10px;")
        self.capture_btn.clicked.connect(self.capture_data)
        self.capture_btn.setEnabled(False) 
        btn_layout.addWidget(self.capture_btn)

        layout.addLayout(btn_layout)
        self.worker = None

    def update_plot(self, raw_chunk, filtered_chunk):
        n_filt = len(filtered_chunk)
        n_raw = len(raw_chunk)
        
        if n_filt > 0 and n_raw > 0:
            # --- 1. 更新 RAW Buffer ---
            # 如果是第一次收到資料，利用長度比例計算 RAW Deque 應該要多長
            if self.raw_deque is None:
                ratio = n_raw / n_filt
                raw_max_len = int(self.plot_buffer_size * ratio)
                self.raw_deque = deque(maxlen=raw_max_len)
            
            # 把原始 bits 塞入隊列 (超過長度會自動擠掉舊的)
            self.raw_deque.extend(raw_chunk)

            # --- 2. 更新繪圖 Buffer (處理過的漂亮波形) ---
            self.plot_buffer = np.roll(self.plot_buffer, -n_filt)
            self.plot_buffer[-n_filt:] = filtered_chunk
            
            # 基線校正 (僅供視覺好看)
            baseline = np.median(self.plot_buffer)
            visual_data = self.plot_buffer - baseline
            
            # 降採樣顯示，降低 CPU 繪圖負擔
            plot_data = visual_data[::self.ds_factor]
            
            self.curve.setData(self.plot_time[:len(plot_data)], plot_data)
            self.pw.setYRange(-0.06, 0.08)

    def capture_data(self):
        """🌟 存檔核心：直接把 raw_deque 倒出來存檔"""
        if self.raw_deque is None or len(self.raw_deque) < self.raw_deque.maxlen * 0.9:
            self.status_label.setText("Status: Buffering... Wait!")
            self.status_label.setStyleSheet("color: #FFC107; font-family: Monospace; font-size: 24px; font-weight: bold;")
            return

        # 把 RAW Deque 轉成 Numpy 陣列
        raw_to_save = np.array(self.raw_deque)

        # 產生帶有時間戳記的檔名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_dir}/raw_ecg_2s_{timestamp}.csv"

        try:
            # 儲存純粹原始資料，不做任何基線扣除
            np.savetxt(filename, raw_to_save, delimiter=',', fmt='%.2f')
            self.status_label.setText(f"Status: Saved RAW {timestamp}.csv")
            self.status_label.setStyleSheet("color: #00FF41; font-family: Monospace; font-size: 24px; font-weight: bold;")
        except Exception as e:
            self.status_label.setText("Status: Save Failed!")
            self.status_label.setStyleSheet("color: #FF0000; font-family: Monospace; font-size: 24px; font-weight: bold;")
            print(f">>> [錯誤] 儲存失敗: {e}")

    def toggle_detection(self):
        if self.worker is None or not self.worker.isRunning():
            self.raw_deque = None # 重新連線時清空 RAW 緩衝
            
            self.worker = SPIWorker()
            self.worker.data_ready.connect(self.update_plot)
            self.worker.start()
            
            self.start_btn.setText("STOP")
            self.start_btn.setStyleSheet("background-color: #DC3545; color: white; font-weight: bold; font-size: 16px; border-radius: 10px;")
            self.capture_btn.setEnabled(True)
            self.status_label.setText("Status: Streaming...")
            self.status_label.setStyleSheet("color: #00FF41; font-family: Monospace; font-size: 24px; font-weight: bold;")
        else:
            self.worker.stop()
            self.worker.wait()
            
            self.start_btn.setText("START MONITORING")
            self.start_btn.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold; font-size: 16px; border-radius: 10px;")
            self.capture_btn.setEnabled(False)
            self.status_label.setText("Status: Idle")
            self.status_label.setStyleSheet("color: #888; font-family: Monospace; font-size: 24px; font-weight: bold;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
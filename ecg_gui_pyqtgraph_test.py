import sys
import time
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
from scipy import signal

# --- 匯入您的自定義模組 ---
try:
    import filter           
except ImportError:
    print(">>> [錯誤] 找不到 filter.py 或 quant.py")

# ==========================================
# 🌟 高效 CSV 模擬讀取執行緒
# ==========================================
class CSVWorker(QThread):
    data_ready = pyqtSignal(np.ndarray)

    def __init__(self, filename="./dataset/ecg_hand.csv"):
        super().__init__()
        self._is_running = True
        self.filename = filename

    def run(self):
        try:
            print(f">>> [系統] 正在預載入 {self.filename}...")
            df = pd.read_csv(self.filename, header=None)
            self.raw_data = df.values.flatten()
            self.comp = np.genfromtxt('./filter_coefficient/coef2.dat').flatten()
            self.fir = np.genfromtxt('./filter_coefficient/coef_fir_v2.dat').flatten()
            print(f">>> [系統] 載入完成，總長度: {len(self.raw_data)}")
        except Exception as e:
            print(f">>> [錯誤] 檔案讀取失敗: {e}")
            return
        
        fs_hw = 24000 
        sos_band = signal.butter(3, [1.0, 35], fs=fs_hw, btype='band', output='sos')
        zi_band = signal.sosfilt_zi(sos_band)

        chunk_step = 16384 
        ptr = 0
        total_len = len(self.raw_data)

        while self._is_running:
            if ptr + chunk_step >= total_len:
                ptr = 0 
            
            bits_chunk = self.raw_data[ptr : ptr + chunk_step]
            ptr += chunk_step

            try:
                # 執行前端濾波
                fir_out, _ = filter.filter(bits_chunk, self.comp, self.fir)
                # IIR 帶通濾波
                final_ecg, zi_band = signal.sosfilt(sos_band, fir_out, zi=zi_band)
                self.data_ready.emit(final_ecg)
            except:
                pass
            
            # 控制播放速度
            time.sleep(0.02)

    def stop(self):
        self._is_running = False

# ==========================================
# 🌟 極簡主界面 (純波形與 R-Peak)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pure ECG & R-Peak Monitor")
        self.resize(1200, 700)
        self.setStyleSheet("background-color: #0D0D0D;") 

        # --- 核心參數 ---
        self.fs = 13410             
        self.buffer_size = 65536    
        self.ds_factor = 4 
        self.current_window_sec = 2 
        
        # 顯示用 Buffer (長度 2 秒)
        self.buffer = np.zeros(self.buffer_size)
        
        # 偵測專用 Buffer (長度 2.5 秒，保護邊界)
        self.det_buffer_len = int(self.fs * 2.5) 
        self.det_buffer = np.zeros(self.det_buffer_len)
        
        self.is_monitoring = False      

        self.setup_ui()
        self.update_axis()

    def setup_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # --- 繪圖區 ---
        self.pw = pg.PlotWidget()
        self.pw.setBackground('#0D0D0D')
        self.pw.showGrid(x=True, y=True, alpha=0.2)
        self.pw.setAntialiasing(False) 
        self.pw.setStyleSheet("border: 1px solid #333; border-radius: 8px;")
        
        # 設定綠色波形與紅色 R-Peak 點點
        self.curve = self.pw.plot(pen=pg.mkPen(color='#00FF41', width=1.5))
        self.scatter = pg.ScatterPlotItem(size=12, pen=None, brush=pg.mkBrush(255, 50, 50), symbol='o')
        self.pw.addItem(self.scatter)
        main_layout.addWidget(self.pw)

        # --- 控制按鈕 ---
        self.start_btn = QPushButton("▶ START SIMULATION")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setStyleSheet("background-color: #0066CC; color: white; font-weight: bold; border-radius: 8px; font-size: 20px;")
        self.start_btn.clicked.connect(self.toggle_detection)
        main_layout.addWidget(self.start_btn)

    def update_axis(self):
        view_points = min(int(self.fs * self.current_window_sec), self.buffer_size)
        self.plot_time = np.linspace(-self.current_window_sec, 0, view_points // self.ds_factor)

    def update_plot(self, data):
        if not self.is_monitoring: return
        
        n = len(data)
        if n > 0:
            # 1. 更新 Buffer
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = data
            self.det_buffer = np.roll(self.det_buffer, -n)
            self.det_buffer[-n:] = data
            
            view_size = int(self.fs * self.current_window_sec)
            raw_view = self.buffer[-view_size:]
            
            # ==========================================
            # 🌟 R-Peak 偵測與校正邏輯 (保留最強防錯設計)
            # ==========================================
            # 去除基線
            clean_sig = self.det_buffer - np.median(self.det_buffer)
            
            # Pan-Tompkins 核心
            diff_ecg = np.diff(clean_sig)
            sq = diff_ecg ** 2
            w = int(self.fs * 0.08)
            mwa = np.convolve(sq, np.ones(w)/w, mode='same')
            
            # 正規化
            mwa_max = np.max(mwa)
            if mwa_max > 0: 
                mwa = mwa / mwa_max
            
            # 尋找候選峰值
            peaks, _ = signal.find_peaks(mwa, height=0.35, distance=int(self.fs * 0.2))
            
            # 絕對值尋峰校正 (確保打在真實的最高/最低點)
            refined_peaks = []
            for p in peaks:
                search = int(self.fs * 0.1)
                s_idx, e_idx = max(0, p-search), min(self.det_buffer_len, p+search)
                if s_idx < e_idx:
                    r_idx = s_idx + np.argmax(np.abs(clean_sig[s_idx:e_idx]))
                    refined_peaks.append(r_idx)
                    
            refined_peaks = np.unique(refined_peaks)

            # 過濾出當前 2 秒畫面內的可見 Peak
            valid_peaks = [p for p in refined_peaks if p > (self.det_buffer_len - view_size)]

            # ==========================================
            # 🌟 繪圖更新
            # ==========================================
            # 畫波形 (降採樣加速)
            plot_data = raw_view[::self.ds_factor]
            self.curve.setData(self.plot_time[:len(plot_data)], plot_data)
            
            # 畫紅點
            if len(valid_peaks) > 0:
                px = (np.array(valid_peaks) - self.det_buffer_len) / self.fs
                py = self.det_buffer[valid_peaks]
                self.scatter.setData(x=px, y=py)
            else:
                self.scatter.setData([], [])

            # Y 軸自適應置中
            self.pw.setYRange(np.min(raw_view)*1.2, np.max(raw_view)*1.2)

    def toggle_detection(self):
        if not self.is_monitoring:
            self.buffer.fill(0)
            self.det_buffer.fill(0)
            self.is_monitoring = True
            
            self.worker = CSVWorker() 
            self.worker.data_ready.connect(self.update_plot)
            self.worker.start()
            
            self.start_btn.setText("⏸ STOP SIMULATION")
            self.start_btn.setStyleSheet("background-color: #CC0000; color: white; font-weight: bold; border-radius: 8px; font-size: 20px;")
        else:
            self.is_monitoring = False
            self.worker.stop()
            self.worker.wait()
            
            self.start_btn.setText("▶ START SIMULATION")
            self.start_btn.setStyleSheet("background-color: #0066CC; color: white; font-weight: bold; border-radius: 8px; font-size: 20px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
import sys
import os
import numpy as np
import statistics
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QLabel, QPushButton, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg
from scipy import signal

try:
    import filter           
    from spi_receive import SerialStreaming
except ImportError:
    print(">>> [錯誤] 找不到 filter.py 或 spi_receive.py")

class SPIWorker(QThread):
    data_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        self.stream = SerialStreaming() 
        if not self.stream.connect(): return
        
        comp = np.genfromtxt('coef2.dat').flatten()
        fir = np.genfromtxt('coef_fir_v2.dat').flatten()
        
        fs_hw = 24000 
        # 針對人體訊號，帶通濾波限縮在 1Hz ~ 35Hz，有效去除呼吸與 60Hz 雜訊
        sos_band = signal.butter(3, [1.0, 35], fs=fs_hw, btype='band', output='sos')
        zi_band = signal.sosfilt_zi(sos_band)

        while self._is_running:
            bits = self.stream.get_chunk()
            if bits is not None:
                try:
                    fir_out, _ = filter.filter(bits, comp, fir)
                    final_ecg, zi_band = signal.sosfilt(sos_band, fir_out, zi=zi_band)
                    self.data_ready.emit(final_ecg)
                except: pass
            else:
                self.msleep(1)

    def stop(self):
        self._is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG R-Peak Monitor (HRV / Fatigue Analysis Ready)")
        self.resize(1100, 800)
        self.setStyleSheet("background-color: #000;")

        # --- 核心參數 ---
        self.fs = 13410             
        self.buffer_size = 65536    
        self.ds_factor = 4          
        self.current_window_sec = 2 
        
        self.buffer = np.zeros(self.buffer_size)
        self.bpm_history = []

        self.setup_ui()
        self.update_axis()

    def setup_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        
        info_layout = QHBoxLayout()
        self.hr_label = QLabel("HR: -- BPM")
        self.hr_label.setStyleSheet("color: #00FF41; font-family: Monospace; font-size: 48px; font-weight: bold;")
        info_layout.addWidget(self.hr_label)
        
        self.debug_label = QLabel("Algorithm: MWA Enveloping")
        self.debug_label.setStyleSheet("color: #888; font-family: Monospace; font-size: 14px;")
        info_layout.addWidget(self.debug_label, alignment=Qt.AlignRight)
        layout.addLayout(info_layout)

        self.pw = pg.PlotWidget()
        self.pw.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.pw.plot(pen=pg.mkPen(color='#00FFFF', width=2))
        
        self.r_peaks_scatter = pg.ScatterPlotItem(size=12, pen=None, brush=pg.mkBrush(255, 0, 0), symbol='x')
        self.pw.addItem(self.r_peaks_scatter)
        layout.addWidget(self.pw)

        self.start_btn = QPushButton("START MONITORING")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold; border-radius: 10px;")
        self.start_btn.clicked.connect(self.toggle_detection)
        layout.addWidget(self.start_btn)

        self.worker = None

    def update_axis(self):
        view_points = min(int(self.fs * self.current_window_sec), self.buffer_size)
        plot_points = view_points // self.ds_factor
        self.plot_time = np.linspace(-view_points/self.fs, 0, plot_points)

    def update_plot(self, data):
        n = len(data)
        if n > 0:
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = data
            
            view_size = min(int(self.fs * self.current_window_sec), self.buffer_size)
            raw_view = self.buffer[-view_size:]
            
            # ==========================================
            # 修正版：加入「正規化」的 Pan-Tompkins 演算法
            # ==========================================
            
            # 1. 微分 (取斜率)
            diff_ecg = np.diff(raw_view)
            
            # 2. 平方 (放大 R 波，但數字會變極小)
            squared_ecg = diff_ecg ** 2
            
            # 3. 移動平均 (縮短視窗到 80ms，避免把小峰值壓扁)
            window_size = int(self.fs * 0.08) 
            mwa_ecg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
            
            # 4. 正規化 (將整個波包的數值拉伸到 0 ~ 1 之間，徹底解決振幅太小的問題)
            mwa_max = np.max(mwa_ecg)
            if mwa_max > 0:
                mwa_ecg = mwa_ecg / mwa_max
            
            # 5. 抓取特徵點 (因為最大值是 1，門檻固定設 0.35 即可)
            feature_indices, _ = signal.find_peaks(mwa_ecg, 
                                                   height=0.35, 
                                                   distance=int(self.fs * 0.2))
            
            # 6. 座標校正 (回到原始訊號找真正的 R 波頂點)
            refined_peaks = []
            search_offset = int(self.fs * 0.05) # 前後搜尋 50ms
            
            for p in feature_indices:
                start = max(0, p - search_offset)
                end = min(len(raw_view), p + search_offset)
                if start < end:
                    real_peak = start + np.argmax(raw_view[start:end])
                    refined_peaks.append(real_peak)
            
            peaks = np.unique(np.array(refined_peaks))
            # ==========================================

            # --- 視覺與基線校正 ---
            baseline = np.median(raw_view)
            raw_zeroed = raw_view - baseline
            plot_data = raw_zeroed[::self.ds_factor]
            self.curve.setData(self.plot_time[:len(plot_data)], plot_data)

            if len(peaks) > 0:
                # 重新標上紅叉叉
                peak_x = (peaks - view_size) / self.fs
                peak_y = raw_zeroed[peaks]
                self.r_peaks_scatter.setData(x=peak_x, y=peak_y)
                
                # BPM 計算
                if len(peaks) >= 2:
                    avg_samples = np.mean(np.diff(peaks))
                    instant_bpm = (60 * self.fs) / avg_samples
                    self.bpm_history.append(instant_bpm)
                    if len(self.bpm_history) > 10: self.bpm_history.pop(0)
                    
                    final_bpm = int(round(statistics.median(self.bpm_history)))
                    self.hr_label.setText(f"HR: {int(final_bpm/2)} BPM")
            else:
                self.r_peaks_scatter.setData(x=[], y=[])

            self.pw.setYRange(-0.06, 0.08)

    def toggle_detection(self):
        if self.worker is None or not self.worker.isRunning():
            self.worker = SPIWorker()
            self.worker.data_ready.connect(self.update_plot)
            self.worker.start()
            self.start_btn.setText("STOP")
            self.start_btn.setStyleSheet("background-color: #DC3545; color: white; font-weight: bold; border-radius: 10px;")
        else:
            self.worker.stop()
            self.worker.wait()
            self.start_btn.setText("START MONITORING")
            self.start_btn.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold; border-radius: 10px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

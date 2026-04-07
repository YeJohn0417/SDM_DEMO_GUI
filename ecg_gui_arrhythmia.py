import sys
import os
import time
import numpy as np
import statistics
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QLabel, QPushButton, QHBoxLayout, QFrame, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg
from scipy import signal

# --- 匯入您的自定義模組 ---
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
        
        try:
            comp = np.genfromtxt('coef2.dat').flatten()
            fir = np.genfromtxt('coef_fir_v2.dat').flatten()
        except Exception as e:
            print(f">>> [錯誤] 檔案讀取失敗: {e}")
            return
        
        fs_hw = 24000 
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
        self.setWindowTitle("Clinical ECG & HRV Analysis System")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #0D0D0D;") # 專業的深淵黑背景

        # --- 核心參數 ---
        self.fs = 13410             
        self.buffer_size = 65536    
        self.ds_factor = 4          
        self.current_window_sec = 2 
        
        self.buffer = np.zeros(self.buffer_size)
        
        # 絕對位置與分析變數
        self.total_samples_received = 0  
        self.absolute_peaks = []         
        self.monitoring_start_time = 0  
        self.is_monitoring = False      
        self.analysis_done = False      

        self.setup_ui()
        self.update_axis()

    def create_data_card(self, title_text, value_text, value_color):
        """建立專業的數據卡片 UI 元件"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                border: 1px solid #333333;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel(title_text)
        title.setStyleSheet("color: #888888; font-family: 'Segoe UI', Arial; font-size: 16px; font-weight: bold; border: none;")
        title.setAlignment(Qt.AlignCenter)
        
        value = QLabel(value_text)
        value.setStyleSheet(f"color: {value_color}; font-family: 'Segoe UI', Arial; font-size: 36px; font-weight: bold; border: none;")
        value.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(value)
        return card, value

    def setup_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # ==========================================
        # 🌟 頂部：專業數據面板 (3 卡片佈局)
        # ==========================================
        dashboard_layout = QHBoxLayout()
        dashboard_layout.setSpacing(20)
        
        # 1. 心率卡片
        self.card_hr, self.val_hr = self.create_data_card("HEART RATE (BPM)", "--", "#00E5FF")
        dashboard_layout.addWidget(self.card_hr)
        
        # 2. 心律不整卡片
        self.card_arr, self.val_arr = self.create_data_card("ARRHYTHMIA (60s)", "Ready", "#777777")
        dashboard_layout.addWidget(self.card_arr)
        
        # 3. HRV 卡片
        self.card_hrv, self.val_hrv = self.create_data_card("HRV METRICS", "RMSSD: -- ms\npNN50: -- %", "#00FF41")
        self.val_hrv.setStyleSheet("color: #00FF41; font-family: 'Segoe UI', Arial; font-size: 22px; font-weight: bold; border: none;")
        dashboard_layout.addWidget(self.card_hrv)
        
        main_layout.addLayout(dashboard_layout)

        # 狀態提示列 (小字)
        self.debug_label = QLabel("System Status: Idle | Ready to start 60s clinical analysis.")
        self.debug_label.setStyleSheet("color: #666666; font-family: 'Segoe UI', Arial; font-size: 14px;")
        self.debug_label.setAlignment(Qt.AlignRight)
        main_layout.addWidget(self.debug_label)

        # ==========================================
        # 🌟 繪圖區 (舒展比例)
        # ==========================================
        self.pw = pg.PlotWidget()
        self.pw.setBackground('#0D0D0D')
        self.pw.showGrid(x=True, y=True, alpha=0.2)
        # 設定邊框讓圖表看起來更精緻
        self.pw.setStyleSheet("border: 1px solid #333333; border-radius: 8px;")
        
        # 改用螢光綠色畫線，更有傳統醫療螢幕的感覺
        self.curve = self.pw.plot(pen=pg.mkPen(color='#00FF41', width=1.5))
        self.r_peaks_scatter = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(255, 50, 50), symbol='o')
        self.pw.addItem(self.r_peaks_scatter)
        main_layout.addWidget(self.pw)

        # ==========================================
        # 🌟 底部控制區 (扁平化現代按鈕)
        # ==========================================
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)
        
        self.start_btn = QPushButton("▶ START 60s ANALYSIS")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066CC; color: white; font-family: 'Segoe UI', Arial; font-size: 16px; font-weight: bold; border-radius: 6px;
            }
            QPushButton:hover { background-color: #0077EE; }
        """)
        self.start_btn.clicked.connect(self.toggle_detection)
        
        self.retest_btn = QPushButton("↺ RETEST")
        self.retest_btn.setMinimumHeight(50)
        self.retest_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333; color: #AAAAAA; font-family: 'Segoe UI', Arial; font-size: 16px; font-weight: bold; border-radius: 6px;
            }
        """)
        self.retest_btn.clicked.connect(self.retest_action)
        self.retest_btn.setEnabled(False) 

        control_layout.addWidget(self.start_btn, 3) # Start 較寬
        control_layout.addWidget(self.retest_btn, 1) # Retest 較窄
        main_layout.addLayout(control_layout)

        self.worker = None

    def update_axis(self):
        view_points = min(int(self.fs * self.current_window_sec), self.buffer_size)
        plot_points = view_points // self.ds_factor
        self.plot_time = np.linspace(-view_points/self.fs, 0, plot_points)

    def update_plot(self, data):
        if not self.is_monitoring: return
        
        n = len(data)
        if n > 0:
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = data
            self.total_samples_received += n
            
            view_size = min(int(self.fs * self.current_window_sec), self.buffer_size)
            raw_view = self.buffer[-view_size:]
            
            # --- Pan-Tompkins ---
            diff_ecg = np.diff(raw_view)
            squared_ecg = diff_ecg ** 2
            window_size = int(self.fs * 0.08) 
            mwa_ecg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
            mwa_max = np.max(mwa_ecg)
            if mwa_max > 0: mwa_ecg = mwa_ecg / mwa_max
            
            feature_indices, _ = signal.find_peaks(mwa_ecg, height=0.35, distance=int(self.fs * 0.3))
            
            refined_peaks = []
            search_offset = int(self.fs * 0.05) 
            for p in feature_indices:
                start = max(0, p - search_offset)
                end = min(len(raw_view), p + search_offset)
                if start < end:
                    real_peak = start + np.argmax(raw_view[start:end])
                    refined_peaks.append(real_peak)
            peaks = np.unique(np.array(refined_peaks))

            # --- 繪製 ---
            baseline = np.median(raw_view)
            raw_zeroed = raw_view - baseline
            plot_data = raw_zeroed[::self.ds_factor]
            self.curve.setData(self.plot_time[:len(plot_data)], plot_data)

            if not self.analysis_done:
                for p in peaks:
                    abs_p = self.total_samples_received - view_size + p
                    if len(self.absolute_peaks) == 0 or (abs_p - self.absolute_peaks[-1]) > int(self.fs * 0.25):
                        self.absolute_peaks.append(abs_p)

            # --- 標示紅點 ---
            if len(peaks) > 0:
                peak_x = (peaks - view_size) / self.fs
                peak_y = raw_zeroed[peaks]
                self.r_peaks_scatter.setData(x=peak_x, y=peak_y)
            else:
                self.r_peaks_scatter.setData(x=[], y=[])

            # ==========================================
            # 60秒鎖定邏輯
            # ==========================================
            current_time = time.time()
            elapsed_time = current_time - self.monitoring_start_time

            if not self.analysis_done:
                if elapsed_time < 60.0:
                    # 更新 BPM
                    if len(self.absolute_peaks) >= 2:
                        rr_intervals = np.diff(self.absolute_peaks) / self.fs
                        recent_rr = rr_intervals[-5:] if len(rr_intervals) >= 5 else rr_intervals
                        bpm = 60.0 / np.mean(recent_rr)
                        self.val_hr.setText(f"{int(bpm/2)}")

                    # 倒數顯示
                    countdown = int(60 - elapsed_time)
                    self.val_arr.setText(f"Wait: {countdown}s")
                    self.val_arr.setStyleSheet("color: #777777; font-family: 'Segoe UI', Arial; font-size: 36px; font-weight: bold; border: none;")
                else:
                    self.perform_final_analysis()
            
            self.pw.setYRange(-0.06, 0.08)

    def perform_final_analysis(self):
        """鎖定數據，計算最終的一分鐘結果"""
        self.analysis_done = True
        
        # 啟用 Retest 按鈕
        self.retest_btn.setEnabled(True) 
        self.retest_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57; color: white; font-family: 'Segoe UI', Arial; font-size: 16px; font-weight: bold; border-radius: 6px;
            }
            QPushButton:hover { background-color: #3CB371; }
        """)

        if len(self.absolute_peaks) >= 5:
            rr_intervals = np.diff(self.absolute_peaks) / self.fs
            rr_diffs = np.abs(np.diff(rr_intervals))
            
            if len(rr_diffs) > 0:
                rmssd = np.sqrt(np.mean(rr_diffs**2)) * 1000
                pnn50 = (np.sum(rr_diffs > 0.05) / len(rr_diffs)) * 100
                
                ectopic_beats = 0
                for i in range(1, len(rr_intervals)):
                    if rr_intervals[i] < 0.8 * rr_intervals[i-1] or rr_intervals[i] > 1.2 * rr_intervals[i-1]:
                        ectopic_beats += 1
                
                ratio = (ectopic_beats / len(rr_intervals)) * 100
                
                self.val_hrv.setText(f"RMSSD: {rmssd:.1f} ms\npNN50: {pnn50:.1f} %")
                self.val_arr.setText(f"{ratio:.1f}%")
                
                if ratio > 10.0:
                    self.val_arr.setStyleSheet("color: #FF3333; font-family: 'Segoe UI', Arial; font-size: 42px; font-weight: bold; border: none;")
                else:
                    self.val_arr.setStyleSheet("color: #FFB300; font-family: 'Segoe UI', Arial; font-size: 42px; font-weight: bold; border: none;")
                
                self.debug_label.setText("System Status: 1-Min Clinical Analysis Complete (Data Locked).")
                self.debug_label.setStyleSheet("color: #00FF41; font-family: 'Segoe UI', Arial; font-size: 14px;")
        else:
            self.val_arr.setText("No Data")
            self.val_arr.setStyleSheet("color: #FF3333; font-family: 'Segoe UI', Arial; font-size: 36px; font-weight: bold; border: none;")

    def retest_action(self):
        """重新測試按鈕：重置狀態"""
        self.absolute_peaks = []
        self.monitoring_start_time = time.time()
        self.analysis_done = False
        
        self.retest_btn.setEnabled(False)
        self.retest_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333; color: #AAAAAA; font-family: 'Segoe UI', Arial; font-size: 16px; font-weight: bold; border-radius: 6px;
            }
        """)
        
        self.val_hr.setText("--")
        self.val_hrv.setText("RMSSD: -- ms\npNN50: -- %")
        self.val_arr.setText("Wait: 60s")
        self.val_arr.setStyleSheet("color: #777777; font-family: 'Segoe UI', Arial; font-size: 36px; font-weight: bold; border: none;")
        
        self.debug_label.setText("System Status: Restarting 60s Analysis...")
        self.debug_label.setStyleSheet("color: #888888; font-family: 'Segoe UI', Arial; font-size: 14px;")

    def toggle_detection(self):
        if self.worker is None or not self.worker.isRunning():
            self.total_samples_received = 0
            self.absolute_peaks = []
            self.monitoring_start_time = time.time()
            self.is_monitoring = True
            self.analysis_done = False
            
            self.worker = SPIWorker()
            self.worker.data_ready.connect(self.update_plot)
            self.worker.start()
            
            self.start_btn.setText("⏸ STOP MONITORING")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #CC0000; color: white; font-family: 'Segoe UI', Arial; font-size: 16px; font-weight: bold; border-radius: 6px;
                }
                QPushButton:hover { background-color: #EE0000; }
            """)
        else:
            self.is_monitoring = False
            self.worker.stop()
            self.worker.wait()
            
            self.start_btn.setText("▶ START 60s ANALYSIS")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0066CC; color: white; font-family: 'Segoe UI', Arial; font-size: 16px; font-weight: bold; border-radius: 6px;
                }
                QPushButton:hover { background-color: #0077EE; }
            """)
            self.retest_btn.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
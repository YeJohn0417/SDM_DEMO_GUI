# SDM_DEMO_GUI

這程式需要搭配 ESP32 或者前端 serial 訊號使用，主要是用來接收 COM 端傳進來的訊號進行濾波，由於 SDM ADC 輸出訊號較為特殊，是需要透過 decimation filter 進行還原，詳細內容可以詢問類比專業人員。


對了，我在dataset內部有放實際接收 ADC 出來訊號的資料(`ecg_hand.csv`)，可以當作參考，要記得看!!!!

---

## 🌟 主要功能
* **整體系統展示**：執行 `ecg_gui_arrhythmia.py`，接上前端 ESP32 或其他 MCU，及時顯示 + 整合其他演算法的 gui，展示用。
* **測試演算法正確性**：執行 `ecg_gui_pyqtgraph_test.py`，程式會重複讀取 `./dataset/ecg_hand.csv` 的資料繪製成圖。
* **數位濾波器原理**：執行 `filter.py`，有數位濾波器的演算法，數位濾波器設計可以用 mathlab 內建的 filter app 進行設計，app內有頻率響應。
* **資料接收**：執行 `spi_receive.py`子程式，接收 COM 進來的訊號。
* **量化**：執行 `quant.py`，對資料量化，為後續作成硬體使用。

---

## 🔗 前端硬體與通訊程式

本專案主要負責後端 PC 的訊號接收、數位濾波與 ECG 波形顯示。若要進行即時訊號量測，需搭配前端 **ESP32 或其他 MCU**，將 SDM ADC 輸出的資料傳輸至 PC。

**前端 ESP32 程式：**

[ADC_transfer_to_PC — GitHub Repository](https://github.com/YeJohn0417/ADC_transfer_to_PC)

前端程式使用 **VS Code + PlatformIO** 進行韌體編譯與燒錄。

**注意事項：**
- 即時量測模式需連接前端 ESP32 或其他相容的 MCU。
- 若沒有實際硬體，可使用 `./dataset/ecg_hand.csv` 測試濾波器與 GUI 顯示功能。
- 執行 `ecg_gui_pyqtgraph_test.py` 可直接讀取測試資料，不需要連接 ESP32。

---

## 💻 執行環境
在 `environment.yml` 寫好執行需要的套件，直接在終端機輸入
```
conda env create -f environment.yml
```
然後輸入以下指令，激活環境
```
conda activate sdm_ecg_env
```
---

## 📁 檔案結構說明
```text
SDM_DEMO_GUI/
├── dataset/                    # 存放原始 ECG 測試資料集，從 SDM ADC 出來訊號
├── filter_coefficient/         # 存放濾波器設計的係數檔案 (.dat)
├── ecg_gui_arrhythmia.py       # 心律不整簡單演算法 + 即時顯示 ECG GUI 介面，需要接上 ESP32 才能執行
├── ecg_gui_r_peak_detection.py # 偵測 r-peak + 即時顯示 ECG GUI 介面，需要接上 ESP32 才能執行
├── ecg_gui_pyqtgraph_test.py   # 讀取CSV檔案畫圖 + 即時顯示 ECG GUI 介面，可以用來測試演算法是否正確
├── filter.py                   # 數位濾波器實作
├── superfilter.py              # fir、iir濾波器實作
├── spi_receive.py              # 可以用來與 ESP32 進行訊號擷取，需要接上 ESP32 才能執行
├── quant.py                    # 資料量化與定點化處理
├── environment.yml             # 執行環境建立
└── README.md                   # 本說明文件

```
**祝大家都順利畢業~~**

import serial
import numpy as np
import time

# --- 參數設定 ---
COM_PORT = 'COM11'
BAUD_RATE = 1000000
CHUNK_SIZE = 16384 * 4  
BYTES_TO_READ = CHUNK_SIZE // 8

class SerialStreaming:
    def __init__(self):
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            print("Cleaning buffer...")
            time.sleep(1)
            self.ser.reset_input_buffer()
            print("Sending Start Command...")
            self.ser.write(bytes([0x09]))
            return True
        except Exception as e:
            print(f"Connect Error: {e}")
            return False

    def get_chunk(self):
        if self.ser and self.ser.is_open:
            raw_bytes = self.ser.read(BYTES_TO_READ)
            if len(raw_bytes) > 0:
                data_array = np.frombuffer(raw_bytes, dtype=np.uint8)
                bits = np.unpackbits(data_array)
                return bits.astype(np.float64) 
        return None

    def close(self):
        if self.ser:
            self.ser.write(b's')
            self.ser.close()
import serial
import time

PORT_arduino = '/dev/ttyUSB0'
PORT_yuyin = '/dev/ttyUSB0'
baud_rate = 9600

class Arduino_Chat:
    def __init__(self):
        self.ser_arduino = serial.Serial(PORT_arduino, baud_rate)

    def uart_beep(self):
        # self.ser_arduino.write(b'BEEP\n')
        try:
            self.ser_arduino.write(b'BEEP\n')
        except serial.SerialException as e:
            print(f"Serial write error: {e}")

class Yuyin_Chat:
    def __init__(self):
        self.ser_yuyin = serial.Serial(PORT_yuyin, baud_rate)

    def uart_beep(self):
        command1 = bytes.fromhex('7E FF 06 03 00 00 01 EF') #指定第一个文件播放
        command2 = bytes.fromhex('7E FF 06 03 00 00 01 EF') #
        try:
            self.ser_yuyin.write(command1)
        except serial.SerialException as e:
            print(f"Serial write error: {e}")


if __name__ == '__main__':
    # arduino_chat = Arduino_Chat()
    # time.sleep(2)
    # arduino_chat.uart_beep()

    yuyin_chat = Yuyin_Chat()
    # time.sleep(2)
    while True:
        yuyin_chat.uart_beep()
        time.sleep(1)

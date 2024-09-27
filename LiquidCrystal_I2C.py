import smbus
from time import sleep

# Lớp giao tiếp I2C
class i2c_device:
    def __init__(self, addr, port=1):
        self.addr = addr
        self.bus = smbus.SMBus(port)

    def write_cmd(self, cmd):
        self.bus.write_byte(self.addr, cmd)
        sleep(0.0001)

    def write_cmd_arg(self, cmd, data):
        self.bus.write_byte_data(self.addr, cmd, data)
        sleep(0.0001)

    def write_block_data(self, cmd, data):
        self.bus.write_block_data(self.addr, cmd, data)
        sleep(0.0001)

    def read(self):
        return self.bus.read_byte(self.addr)

    def read_data(self, cmd):
        return self.bus.read_byte_data(self.addr, cmd)

    def read_block_data(self, cmd):
        return self.bus.read_block_data(self.addr, cmd)

# Địa chỉ LCD và các lệnh
LCD_ADDRESS = 0x27
LCD_CLEARDISPLAY = 0x01
LCD_RETURNHOME = 0x02
LCD_ENTRYMODESET = 0x04
LCD_DISPLAYCONTROL = 0x08
LCD_FUNCTIONSET = 0x20
LCD_BACKLIGHT = 0x08
En = 0b00000100  # Bit Enable
Rs = 0b00000001  # Bit Register Select

class lcd:
    def __init__(self, address=LCD_ADDRESS):
        self.lcd_device = i2c_device(address)
        self.init_lcd()

    # Khởi tạo màn hình LCD
    def init_lcd(self):
        self.lcd_write(0x03)
        self.lcd_write(0x03)
        self.lcd_write(0x03)
        self.lcd_write(0x02)
        self.lcd_write(LCD_FUNCTIONSET | 0x08)  # 2 dòng, 5x8 điểm
        self.lcd_write(LCD_DISPLAYCONTROL | 0x04)  # Bật màn hình
        self.clear()
        self.lcd_write(LCD_ENTRYMODESET | 0x02)  # Mặc định hướng nhập từ trái sang phải
        sleep(0.2)

    # Gửi tín hiệu EN để chốt lệnh
    def lcd_strobe(self, data):
        self.lcd_device.write_cmd(data | En | LCD_BACKLIGHT)
        sleep(0.0005)
        self.lcd_device.write_cmd((data & ~En) | LCD_BACKLIGHT)
        sleep(0.0001)

    def lcd_write_four_bits(self, data):
        self.lcd_device.write_cmd(data | LCD_BACKLIGHT)
        self.lcd_strobe(data)

    # Ghi lệnh hoặc dữ liệu đến LCD
    def lcd_write(self, cmd, mode=0):
        self.lcd_write_four_bits(mode | (cmd & 0xF0))
        self.lcd_write_four_bits(mode | ((cmd << 4) & 0xF0))

    # Ghi một ký tự ra màn hình LCD
    def lcd_write_char(self, charvalue):
        self.lcd_write(ord(charvalue), Rs)

    # Hiển thị chuỗi ký tự lên một dòng
    def display_line(self, string, line):
        if line == 1:
            self.lcd_write(0x80)  # Địa chỉ của dòng 1
        elif line == 2:
            self.lcd_write(0xC0)  # Địa chỉ của dòng 2
        elif line == 3:
            self.lcd_write(0x94)  # Địa chỉ của dòng 3
        elif line == 4:
            self.lcd_write(0xD4)  # Địa chỉ của dòng 4
        for char in string:
            self.lcd_write_char(char)

    # Xóa màn hình LCD
    def clear(self):
        self.lcd_write(LCD_CLEARDISPLAY)
        self.lcd_write(LCD_RETURNHOME)

    # Điều khiển đèn nền LCD (bật = 1, tắt = 0)
    def backlight(self, state):
        if state == 1:
            self.lcd_device.write_cmd(LCD_BACKLIGHT)
        else:
            self.lcd_device.write_cmd(0x00)

    # Hiển thị chuỗi với vị trí cụ thể
    def display(self, string, line, pos):
        pos_map = [0x80, 0xC0, 0x94, 0xD4]
        if 1 <= line <= 4:
            self.lcd_write(pos_map[line - 1] + pos)
            for char in string:
                self.lcd_write_char(char)

# Ví dụ sử dụng:
# lcd_screen = lcd()
# lcd_screen.display_line("Hello World", 1)
# lcd_screen.display("Raspberry Pi", 2, 5)

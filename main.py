import cv2
import threading
import numpy as np
from ultralytics import YOLO
import LiquidCrystal_I2C
import time

# Khởi tạo màn hình LCD
lcd_screen = LiquidCrystal_I2C.lcd()

# Giảm độ phân giải
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

# Khởi tạo mô hình YOLOv8n (nano)
model = YOLO("yolov8n.pt")

# Cập nhật màn hình LCD
def update_lcd(detections):
    if len(detections) == 0:
        lcd_screen.display("No Detection", 0, 0)
    else:
        object_name = model.model.names[detections[0][-2]]
        lcd_screen.display(object_name, 0, 0)
        print(f"Detected: {object_name}")

# Xử lý webcam trong luồng riêng
def process_frame(frame, frame_width, frame_height):
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    update_lcd(detections)

# Hàm chính
def main():
    frame_width, frame_height = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize để giảm tải cho xử lý
        frame_resized = cv2.resize(frame, (320, 240))

        # Khởi tạo luồng xử lý mô hình YOLO
        threading.Thread(target=process_frame, args=(frame_resized, 320, 240)).start()

        # Hiển thị khung hình lên cửa sổ
        cv2.imshow("YOLOv8", frame_resized)

        # Thoát vòng lặp nếu nhấn phím ESC
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

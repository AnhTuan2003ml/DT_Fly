import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import LiquidCrystal_I2C

# Khởi tạo màn hình LCD
lcd_screen = LiquidCrystal_I2C.lcd()

# Định nghĩa vùng đa giác để xác định khu vực quan tâm
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

# Xóa màn hình LCD
lcd_screen.clear()

# Hàm để phân tích các tham số dòng lệnh
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[320, 240],  # Đặt độ phân giải thấp hơn để tăng hiệu suất
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

# Hàm chính thực hiện quá trình phát hiện đối tượng
def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Thiết lập kết nối với webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Tải mô hình YOLO từ file best.pt
    model = YOLO("best.pt")

    # Tạo vùng đa giác với độ phân giải của webcam
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))

    # Vòng lặp chính để xử lý khung hình từ webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Nếu không đọc được khung hình, thoát khỏi vòng lặp

        # Sử dụng mô hình YOLO để phát hiện đối tượng trong khung hình
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        # Kiểm tra và hiển thị thông tin trên màn hình LCD
        if len(detections) == 0:
            lcd_screen.display("No Detection", 0, 0)  # Dòng 0, vị trí 0
        else:
            for _, confidence, class_id, _ in detections:
                object_name = model.model.names[class_id]
                lcd_screen.display(object_name, 0, 0)  # Hiển thị trên dòng 0
                print(f"Detected: {object_name}")
                break  # Thoát sau khi phát hiện đối tượng đầu tiên

        # Hiển thị khung hình trực tiếp
        cv2.imshow("yolov8", frame)

        # Thoát vòng lặp nếu nhấn phím ESC
        if cv2.waitKey(1) == 27:
            break

    # Giải phóng webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

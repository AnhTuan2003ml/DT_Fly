import cv2  # Thư viện OpenCV để xử lý ảnh và video
import argparse  # Thư viện để phân tích các tham số dòng lệnh

from ultralytics import YOLO  # Thư viện từ Ultralytics để sử dụng mô hình YOLOv8
import supervision as sv  # Thư viện để chú thích và giám sát các đối tượng
import numpy as np  # Thư viện để xử lý các mảng số học
import LiquidCrystal_I2C  # Thư viện để điều khiển màn hình LCD

# Khởi tạo màn hình LCD
lcd_screen = LiquidCrystal_I2C.lcd()  # Sử dụng lớp lcd đã tạo trước

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
        default=[640, 480],  # Giảm độ phân giải xuống 640x480
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

    # Thiết lập công cụ chú thích hộp bao quanh đối tượng
    box_annotator = sv.BoxAnnotator(thickness=2)

    # Tạo vùng đa giác với độ phân giải của webcam
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)

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
            lcd_screen.display("No Detection")
        else:
            # Duyệt qua các đối tượng phát hiện được
            for _, confidence, class_id, _ in detections:
                object_name = model.model.names[class_id]
                # Hiển thị tên đối tượng trên màn hình LCD
                lcd_screen.display(object_name)
                print(f"Detected: {object_name}")  # In ra console
                break  # Thoát sau khi phát hiện đối tượng đầu tiên

        # Tạo nhãn cho các đối tượng phát hiện được
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]
        # Chú thích khung hình với các hộp bao quanh và nhãn
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # Hiển thị khung hình đã chú thích
        cv2.imshow("yolov8", frame)

        # Thoát vòng lặp nếu nhấn phím ESC
        if (cv2.waitKey(30) == 27):
            break

    # Giải phóng webcam
    cap.release()

# Khởi động chương trình khi được chạy
if __name__ == "__main__":
    main()

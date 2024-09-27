import cv2  # Thư viện OpenCV để xử lý ảnh và video
import argparse  # Thư viện để phân tích các tham số dòng lệnh

from ultralytics import YOLO  # Thư viện từ Ultralytics để sử dụng mô hình YOLOv8
import supervision as sv  # Thư viện để chú thích và giám sát các đối tượng
import numpy as np  # Thư viện để xử lý các mảng số học
import LiquidCrystal_I2C  # Thư viện cho màn hình LCD
lcd_screen = LiquidCrystal_I2C.lcd()  # Sử dụng lớp lcd đã tạo trước
import time  # Thư viện để xử lý thời gian

# Định nghĩa vùng đa giác sẽ được sử dụng để xác định khu vực quan tâm trong khung hình
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
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

# Hàm chính thực hiện quá trình phát hiện đối tượng
def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")  # Tải mô hình YOLO từ file best.pt

    box_annotator = sv.BoxAnnotator(
        thickness=2,
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Kiểm tra xem có đọc khung hình không

        result = model(frame, agnostic_nms=True)[0]

        # Lấy các phát hiện từ kết quả
        detections = sv.Detections.from_yolov8(result.boxes.cpu().numpy())

        if len(detections) == 0:
            lcd_screen.display("Save")
        else:
            for _, confidence, class_id, _ in detections:
                if model.model.names[class_id]:
                    lcd_screen.display(model.model.names[class_id])
                else:
                    lcd_screen.display("Non")
                    break

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]
        
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):  # Thoát vòng lặp nếu nhấn phím ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Khởi động chương trình khi được chạy
if __name__ == "__main__":
    main()

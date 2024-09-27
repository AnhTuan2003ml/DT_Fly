import cv2  # Thư viện OpenCV để xử lý ảnh và video
import argparse  # Thư viện để phân tích các tham số dòng lệnh
from ultralytics import YOLO  # Thư viện từ Ultralytics để sử dụng mô hình YOLOv8
import supervision as sv  # Thư viện để chú thích và giám sát các đối tượng
import numpy as np  # Thư viện để xử lý các mảng số học
import LiquidCrystal_I2C 

# Khởi tạo màn hình LCD
lcd_screen = LiquidCrystal_I2C.lcd()  

# Định nghĩa vùng đa giác cho khu vực quan tâm
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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")  # Tải mô hình YOLO từ file best.pt

    # Thiết lập công cụ chú thích hộp bao quanh đối tượng
    box_annotator = sv.BoxAnnotator(thickness=2)

    # Tạo vùng đa giác với độ phân giải của webcam
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)

    # Thiết lập công cụ chú thích vùng đa giác
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sử dụng mô hình YOLO để phát hiện đối tượng trong khung hình
        result = model(frame, agnostic_nms=True)[0]

        # Lấy các đối tượng phát hiện
        boxes = result.boxes  # Lấy các hộp bao quanh
        detections = sv.Detections.from_yolov5(boxes)

        # Nếu không phát hiện đối tượng
        if boxes.data.size(0) == 0:
            lcd_screen.display("Save")
        else:
            # Duyệt qua các đối tượng phát hiện được
            for box in boxes.data:
                confidence = box[4].item()  # Lấy độ tin cậy
                class_id = int(box[5].item())  # Lấy ID lớp
                if class_id < len(model.model.names):
                    lcd_screen.display(model.model.names[class_id])
                else:
                    lcd_screen.display("Unknown")
                break  # Nếu chỉ muốn hiển thị một đối tượng

        # Tạo nhãn cho các đối tượng phát hiện
        labels = [
            f"{model.model.names[int(box[5].item())]} {box[4].item():0.2f}"
            for box in boxes.data if box[4].item() > 0.5  # Lọc theo độ tin cậy
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
    cv2.destroyAllWindows()  # Giải phóng tất cả các cửa sổ OpenCV

# Khởi động chương trình khi được chạy
if __name__ == "__main__":
    main()

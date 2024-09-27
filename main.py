import cv2 # Thư viện OpenCV để xử lý ảnh và video
import argparse # Thư viện để phân tích các tham số dòng lệnh

from ultralytics import YOLO # Thư viện từ Ultralytics để sử dụng mô hình YOLOv8
import supervision as sv # Thư viện để chú thích và giám sát các đối tượng
import numpy as np # Thư viện để xử lý các mảng số học

import serial # Thư viện để giao tiếp qua cổng serial
import time # Thư viện để xử lý thời gian

# Tạo đối tượng Serial để kết nối với cổng COM5 với tốc độ baud 9600
s = serial.Serial('COM5', 9600)

# Định nghĩa vùng đa giác mà sẽ được sử dụng để xác định khu vực quan tâm trong khung hình
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

# Hàm để phân tích các tham số dòng lệnh
def parse_arguments() -> argparse.Namespace:
    # Tạo đối tượng ArgumentParser
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    # Thêm tham số để nhận độ phân giải của webcam từ dòng lệnh
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    # Phân tích các tham số dòng lệnh
    args = parser.parse_args()
    return args

# Hàm chính thực hiện quá trình phát hiện đối tượng
def main():
    # Nhận các tham số dòng lệnh
    args = parse_arguments()
    # Lấy độ phân giải khung hình từ tham số dòng lệnh
    frame_width, frame_height = args.webcam_resolution

    # Thiết lập kết nối với webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Tải mô hình YOLO từ file best.pt
    model = YOLO("best.pt")

    # Thiết lập công cụ chú thích hộp bao quanh đối tượng
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Tạo vùng đa giác với độ phân giải của webcam
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    # Thiết lập vùng đa giác
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    # Thiết lập công cụ chú thích vùng đa giác
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Vòng lặp chính để xử lý khung hình từ webcam
    while True:
        # Đọc khung hình từ webcam
        ret, frame = cap.read()

        # Sử dụng mô hình YOLO để phát hiện đối tượng trong khung hình
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        # Nếu không phát hiện đối tượng, gửi tín hiệu 'd' qua cổng serial
        if len(detections) == 0:
            s.write(b'd')
        else:
            # Duyệt qua các đối tượng phát hiện được
            for _, confidence, class_id, _ in detections:
                if model.model.names[class_id]:
                    # Nếu đối tượng là 'bird', gửi tín hiệu 'a' qua cổng serial
                    if model.model.names[class_id] == 'bird':
                        s.write(b'a')
                        break
                    # Nếu đối tượng là 'drone', gửi tín hiệu 'b' qua cổng serial
                    elif model.model.names[class_id] == 'drone':
                        s.write(b'b')
                        break
                    # Nếu đối tượng khác, gửi tín hiệu 'c' qua cổng serial
                    else:
                        s.write(b'c')
                        break

        # Tạo nhãn cho các đối tượng phát hiện được
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
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

# Khởi động chương trình khi được chạy
if __name__ == "__main__":
    main()

import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load mô hình best.pt
model = YOLO("best.pt")

# 2. Mở Camera (0 là webcam mặc định)
cap = cv2.VideoCapture(0)

# Cấu hình độ phân giải camera HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Không thể mở Camera! Hãy kiểm tra lại thiết bị.")
    exit()

print("Đang mở Camera thành công!")
print("-> Đưa hình biển báo trước Camera và giữ yên 1-2 giây.")
print("-> Bấm phím 'q' trên bàn phím để đóng cửa sổ Camera.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận luồng hình ảnh từ Camera.")
        break

    # 3. Dự đoán với conf=0.15 (bắt siêu nhạy)
    results = model.predict(source=frame, conf=0.15, stream=True, verbose=False)

    for r in results:
        # Kiểm tra xem có phát hiện OBB nào không
        if r.obb is not None and len(r.obb) > 0:
            # Lấy tọa độ 4 góc của từng khung OBB
            xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
            classes = r.obb.cls.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()

            for points, cls_id, conf in zip(xyxyxyxy, classes, confs):
                # Chuyển tọa độ 4 góc về dạng số nguyên
                pts = points.reshape((-1, 1, 2)).astype(np.int32)
                
                # Tên nhãn biển báo và % độ tin cậy
                label = f"{model.names[int(cls_id)]} {conf*100:.0f}%"

                # 1. Vẽ khung đa giác xoay (OBB) màu xanh lá
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

                # 2. Lấy vị trí góc trên cùng của khung để ghi chữ
                top_point = tuple(pts[0][0])
                text_x, text_y = top_point[0], max(top_point[1] - 10, 25)

                # 3. Vẽ nền màu đen đằng sau chữ để dễ nhìn
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (text_x, text_y - text_h - 5), (text_x + text_w, text_y + 5), (0, 0, 0), -1)

                # 4. In trực tiếp tên biển báo + % màu vàng nổi bật
                cv2.putText(frame, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # 4. Hiển thị camera
    cv2.imshow("Nhan Dien Bien Bao Giao Thong - Realtime Camera", frame)

    # Nhấn phím 'q' để đóng camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
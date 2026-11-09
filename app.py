import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# 1. Từ điển ánh xạ Mã biển báo -> Tên tiếng Việt đầy đủ
SIGN_NAMES = {
    "P-106a": "Cấm xe tải",
    "P-106b": "Cấm xe tải trên ngưỡng",
    "P-131a": "Cấm đỗ xe",
    "P-131b": "Cấm đỗ xe ngày lẻ",
    "P-131c": "Cấm đỗ xe ngày chẵn",
    "P-123a": "Cấm rẽ trái",
    "P-123b": "Cấm rẽ phải",
    "P-124a": "Cấm quay đầu",
    "P-130": "Cấm dừng và đỗ xe",
    "R-434": "Bến xe buýt",
    "W-201a": "Chỗ ngoặt nguy hiểm bên trái",
    "W-201b": "Chỗ ngoặt nguy hiểm bên phải",
    "DP-135": "Hết tất cả các lệnh cấm",
    "P-102": "Cấm đi ngược chiều",
    "P-103a": "Cấm ô tô",
    "P-103b": "Cấm ô tô rẽ phải",
    "P-103c": "Cấm ô tô rẽ trái",
    "P-104": "Cấm mô tô",
    "P-107a": "Cấm ô tô khách và ô tô tải",
    "P-112": "Cấm người đi bộ",
    "P-115": "Hạn chế trọng lượng xe",
    "P-117": "Hạn chế chiều cao",
    "P-127": "Tốc độ tối đa cho phép",
    "P-128": "Cấm sử dụng còi",
    "P-124b": "Cấm ô tô quay đầu",
    "P-124c": "Cấm rẽ trái và quay đầu",
    "P-137": "Cấm đi thẳng và rẽ trái",
    "P-245a": "Đi chậm",
    "R-301c": "Hướng đi phải theo: đi thẳng hoặc rẽ trái",
    "R-301d": "Hướng đi phải theo: đi thẳng hoặc rẽ phải",
    "R-301e": "Hướng đi phải theo: đi thẳng hoặc rẽ trái",
    "R-302a": "Hướng phải đi vòng sang bên phải",
    "R-302b": "Hướng phải đi vòng sang bên trái",
    "R-303": "Nơi giao nhau chạy theo vòng xuyến",
    "R-407a": "Đường một chiều",
    "R-409": "Đường dành cho xe đạp",
    "R-425": "Bệnh viện",
    "S-509a": "Chiều cao an toàn",
    "W-202a": "Nhiều chỗ ngoặt nguy hiểm, chỗ ngoặt đầu tiên sang trái",
    "W-202b": "Nhiều chỗ ngoặt nguy hiểm, chỗ ngoặt đầu tiên sang phải",
    "W-203b": "Đường bị hẹp về phía bên phải",
    "W-203c": "Đường bị hẹp về phía bên trái",
    "W-205a": "Giao nhau cùng cấp",
    "W-205b": "Giao nhau cùng cấp (dạng nhánh)",
    "W-205d": "Giao nhau cùng cấp (dạng nhánh)",
    "W-207a": "Giao nhau với đường không ưu tiên",
    "W-207b": "Giao nhau với đường không ưu tiên",
    "W-207c": "Giao nhau với đường không ưu tiên",
    "W-208": "Giao nhau với đường ưu tiên",
    "W-209": "Giao nhau có tín hiệu đèn",
    "W-210": "Giao nhau với đường sắt có rào chắn",
    "W-219": "Dốc xuống nguy hiểm",
    "W-224": "Người đi bộ cắt ngang",
    "W-225": "Trẻ em",
    "W-227": "Công trường",
    "W-233": "Nguy hiểm khác",
    "W-235": "Đường đôi",
    "W-245a": "Đi chậm",
}

st.set_page_config(page_title="Nhận Diện Biển Báo Giao Thông", layout="wide")
st.title("🚦 Hệ Thống Nhận Diện Biển Báo Giao Thông (YOLOv8-OBB)")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.sidebar.header("Cấu hình mô hình")
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.05, 1.0, 0.15, 0.05)
source_type = st.sidebar.radio("Chọn nguồn dữ liệu:", ("Webcam Trực Tiếp", "Tải Ảnh Lên"))

# --- XỬ LÝ WEBCAM TRỰC TIẾP ---
if source_type == "Webcam Trực Tiếp":
    st.subheader("📷 Nhận diện thời gian thực qua Webcam")
    run_cam = st.checkbox("Bật Camera")
    
    FRAME_WINDOW = st.image([])
    
    if run_cam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể kết nối với Camera!")
                break
            
            results = model.predict(source=frame, conf=conf_threshold, stream=True, verbose=False)
            
            for r in results:
                if r.obb is not None and len(r.obb) > 0:
                    xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
                    classes = r.obb.cls.cpu().numpy()
                    confs = r.obb.conf.cpu().numpy()
                    
                    for points, cls_id, conf in zip(xyxyxyxy, classes, confs):
                        pts = points.reshape((-1, 1, 2)).astype(np.int32)
                        
                        # Lấy mã biển báo và tra cứu tên đầy đủ
                        raw_name = model.names[int(cls_id)]
                        full_name = SIGN_NAMES.get(raw_name, "")
                        
                        if full_name:
                            label = f"{raw_name}: {full_name} ({conf*100:.0f}%)"
                        else:
                            label = f"{raw_name} ({conf*100:.0f}%)"
                        
                        # Vẽ khung xoay OBB
                        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                        
                        # Vẽ nền đen + chữ màu vàng
                        top_point = tuple(pts[0][0])
                        text_x, text_y = top_point[0], max(top_point[1] - 10, 25)
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (text_x, text_y - text_h - 5), (text_x + text_w, text_y + 5), (0, 0, 0), -1)
                        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
        cap.release()

# --- XỬ LÝ TẢI ẢNH LÊN ---
elif source_type == "Tải Ảnh Lên":
    st.subheader("🖼️ Nhận diện biển báo từ Tệp Ảnh")
    uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        results = model.predict(source=image, conf=conf_threshold)
        
        for r in results:
            if r.obb is not None and len(r.obb) > 0:
                xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
                classes = r.obb.cls.cpu().numpy()
                confs = r.obb.conf.cpu().numpy()
                
                for points, cls_id, conf in zip(xyxyxyxy, classes, confs):
                    pts = points.reshape((-1, 1, 2)).astype(np.int32)
                    
                    raw_name = model.names[int(cls_id)]
                    full_name = SIGN_NAMES.get(raw_name, "")
                    
                    if full_name:
                        label = f"{raw_name}: {full_name} ({conf*100:.0f}%)"
                    else:
                        label = f"{raw_name} ({conf*100:.0f}%)"
                    
                    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                    
                    top_point = tuple(pts[0][0])
                    text_x, text_y = top_point[0], max(top_point[1] - 10, 25)
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(image, (text_x, text_y - text_h - 5), (text_x + text_w, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Kết quả nhận diện", use_container_width=True)
# BizOCR Python Vision, OCR & Text Normalization Microservice

Microservice chuyên biệt trong hệ thống BizOCR, đảm nhiệm quy trình chuyển đổi thị giác máy tính và ngôn ngữ:
$$\text{Ảnh gốc} \xrightarrow{\text{OpenCV}} \text{Ảnh sạch} \xrightarrow{\text{PaddleOCR}} \text{OCR thô} \xrightarrow{\text{qwen2.5:3b}} \text{Văn bản chuẩn} \xrightarrow{\text{Java Core}} \begin{cases} \text{Regex / Template Matching} \\ \text{AI Fallback (3.5-instruct / Qwen)} \end{cases}$$

---

## 🚀 Các Tầng Xử Lý (Pipeline Architecture)

1. **Tầng 1: Tiền xử lý ảnh OpenCV (`ImageCleaner`)**
   - **Tự động xoay thẳng (Auto Deskew):** Nhận diện góc nghiêng bằng `cv2.minAreaRect` trên ma trận điểm văn bản và xoay thẳng với `cv2.warpAffine`.
   - **Khử bóng & Cân bằng sáng (Shadow Removal & CLAHE):** Khử bóng đổ không đều bằng phép biến đổi hình thái học (Morphological Closing) và cân bằng độ tương phản thích ứng `cv2.createCLAHE`.
   - **Khử nhiễu & Làm nét (Denoise & Sharpen):** Lọc song phương `cv2.bilateralFilter` bảo toàn cạnh ký tự kết hợp nhân tích chập sắc nét viền chữ.

2. **Tầng 2: Nhận diện ký tự quang học PaddleOCR (`PaddleOcrEngine`)**
   - Hỗ trợ `PaddleOCR` (v4/server với từ điển Tiếng Việt + Latin), fallback `RapidOCR`.
   - Thuật toán sắp xếp thứ tự đọc tự nhiên (`Spatial Reading Order`): Gom nhóm các hộp chữ cùng dòng theo trục Y và sắp xếp trái qua phải theo trục X.
   - Trả về Bounding Boxes, Text thô và Confidence Score.

3. **Tầng 3: Chuẩn hóa ngôn ngữ bằng Qwen2.5:3B (`QwenTextNormalizer`)**
   - Tích hợp Ollama LLM (`qwen2.5:3b` / `qwen2.5:3b-instruct`).
   - Sửa lỗi chính tả tiếng Việt do scan/mờ, phục hồi dấu thanh, chuẩn hóa khoảng trắng.
   - **Bảo toàn 100% số liệu:** Mã số thuế (MST), Số tiền, Ngày tháng, Số hóa đơn, Mã khách hàng.
   - Có cơ chế Rule-based Fallback khi LLM tạm thời bận.

4. **Tầng 4: Machine Learning & Analytics (`DocumentMlService`)**
   - Phân loại tài liệu bằng TF-IDF + Logistic Regression (`/api/v1/ml/predict`).
   - Tự động thu thập mẫu huấn luyện từ phê duyệt kế toán (`/api/v1/ml/collect`).
   - Gom cụm tài liệu chưa rõ bằng K-Means (`/api/v1/ml/cluster-unknowns`).
   - Phát hiện số tiền bất thường bằng thuật toán Z-Score / 3-sigma (`/api/v1/ml/analyze-anomaly`).

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy

### 1. Chạy Local bằng Python
```bash
cd python_service
pip install -r requirements.txt
python main.py
```
Dịch vụ sẽ lắng nghe tại `http://localhost:8000`.

### 2. Chạy với Docker
```bash
cd python_service
docker build -t bizocr-python-service .
docker run -d -p 8000:8000 --name bizocr-vision bizocr-python-service
```

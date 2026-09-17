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

# Model B — Guardrailed MLOps Pipeline
## Pipeline order
```text
export_sanitized_dataset.py
validate_and_split.py
train_lora.py
evaluate_gate.py
register_model.py
inference_guardrail.py
```
Run every step with an explicit path.  The scripts intentionally do not
auto-select the newest directory, because that can train or deploy the wrong
artifact in CI.
## Dependencies
Dataset export:
```bash
pip install mysql-connector-python
```
Validation and schema checks:
```bash
pip install jsonschema
```
Training/evaluation:
```bash
pip install torch transformers datasets peft accelerate
```
## Database environment
```bash
export DB_HOST=localhost
export DB_PORT=3306
export DB_USER=model_b_readonly
export DB_PASSWORD='...'
export DB_NAME=ocr_db
```
The default SQL expects `documents` and `field_correction_audits` with aliases
documented in `export_sanitized_dataset.py`.  Use `--sql-file` if the deployed
schema uses different names.
## Example run
```bash
python training_pipeline/export_sanitized_dataset.py \
  --output-root datasets \
  --dataset-version ds-001
python training_pipeline/validate_and_split.py \
  --dataset datasets/ds-001 \
  --schema-dir schemas \
  --min-golden 10
python training_pipeline/train_lora.py \
  --dataset datasets/ds-001 \
  --output-root saved_models \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --load-in-4bit
python training_pipeline/evaluate_gate.py \
  --candidate saved_models/<candidate-version> \
  --schema-dir schemas \
  --min-field-accuracy 0.90 \
  --min-schema-rate 0.99
python training_pipeline/register_model.py \
  --candidate saved_models/<candidate-version>
```
## Operational rules
- Never put database passwords in source code.
- Do not use `--allow-small` outside a local smoke test.
- The golden set must remain isolated from training.
- Evaluation must load the candidate adapter; it must never use the target as
  the prediction.
- Registration updates an atomic pointer and preserves the old candidate.
- A guardrail failure returns `NEEDS_REVIEW`; callers must not persist it as an
  approved extraction.
- The generic guardrail only performs conservative business checks.  Put
  document-specific rules in JSON Schema or a dedicated validator.


# BizOCR — Tổng quan dự án

## 1. Mục tiêu & phạm vi

BizOCR là hệ thống xử lý chứng từ đa tenant (multi-tenant) cho bối cảnh kế toán/doanh nghiệp Việt Nam, tự động hóa: OCR, phân loại chứng từ, trích xuất trường dữ liệu, xác thực công ty, và phát hiện bất thường.

**Loại chứng từ hỗ trợ:** hóa đơn GTGT (VAT), hợp đồng, namecard, hóa đơn điện nước, sao kê ngân hàng, phiếu nội bộ.

**Mục tiêu tổng thể:** một pipeline sẵn sàng production, có quy trình duyệt tay (human-in-the-loop) và cách ly dữ liệu đa tenant qua JWT.

---

## 2. Kiến trúc

**Backend Java:** Spring Boot 3, WebFlux/R2DBC (reactive), MariaDB, hàng đợi bất đồng bộ Redis, Ollama (LLM local, `qwen2.5:3b-instruct`) làm fallback trích xuất trường.

**Python service:** FastAPI (port 8000) — OCR, phân loại ML, trích xuất trường.

**Pipeline gốc:** Upload → OCR → Template Matching → Phân loại ML (Python) → Ollama LLM fallback → Trích xuất trường → Duyệt tay → Approval kèm tra cứu công ty.

**Frontend:** `index.html` console test UI (quy trình 7 bước), thay Postman.

**Tích hợp ngoài:** VietQR API (tra MST), DuckDuckGo Search (đã thay Google Custom Search do Google ngừng cấp dịch vụ tìm kiếm cho project mới), Ollama (chạy CPU-only do xung đột driver GPU).

---

## 3. Công nghệ sử dụng

- **Ngôn ngữ/framework:** Java 21, Spring Boot 3 (WebFlux, R2DBC), Python 3, FastAPI, Project Reactor
- **OCR:** kiến trúc **Hybrid** — PaddleOCR chỉ làm Detection (khoanh vùng chữ, `lang="en"` vì không đọc nội dung), **VietOCR** (`vgg_transformer`) làm Recognition (đọc chữ tiếng Việt có dấu)
- **Database:** MariaDB (R2DBC), Redis
- **AI/ML:** Ollama (`qwen2.5:3b-instruct`, CPU-only), scikit-learn, Python ML classifier
- **APIs ngoài:** VietQR API, DuckDuckGo Search (qua Python microservice riêng, không cần API key/cx)

---

## 4. Lịch sử các vấn đề đã giải quyết

### 4.1 OCR & xử lý ảnh
- Nâng cấp OCR: Tesseract/RapidOCR → PaddleOCR → kiến trúc Hybrid (PaddleOCR detect + VietOCR recognize), giải quyết lỗi dấu tiếng Việt hệ thống (`"hòa" → "höa"`) từ model Latin dùng chung của PaddleOCR
- Sửa bug downscale ảnh: `ImageCleaner.resize_if_needed` mặc định `max_dim=2500`, nâng lên `4000` và khai báo rõ ràng
- Sửa lỗi field email, lỗi API tra MST (chuyển từ Google Custom Search sang DuckDuckGo Search do Google ngừng dịch vụ)
- Sửa lỗi port mismatch: `ExternalTaxSearchService` gọi nhầm port 8090 thay vì 8000 (port thật của Python service)
- Pipeline end-to-end đạt 90–95% kết quả đạt yêu cầu (test trên namecard, Document ID 2050)

### 4.2 Schema & đa tenant (IDOR)
- Tách `owner_company_id` (tenant sở hữu, gán 1 lần lúc upload) khỏi `company_id` (công ty được OCR nhận diện trong nội dung) — trước đó dùng chung 1 cột khiến `PipelineOrchestrator` ghi đè giá trị tenant, gây lỗi 404 sau khi document xử lý xong
- Hợp nhất 3 luồng resolve công ty rời rạc thành 1 pipeline fallback duy nhất trong `CompanyResolverService` (VietQR → DuckDuckGo Search → Ollama)

### 4.3 Xác thực & phân quyền (mới hoàn thành)
- Phát hiện: `AuthController` ban đầu chỉ là stub hardcode 1 tài khoản (`admin`/`admin`), role/companyId gắn cứng; `PasswordEncoder` (BCrypt) được khai báo nhưng không dùng
- Xây dựng hoàn chỉnh: bảng `users` thật trong MariaDB, đăng ký công khai (username/password/email/phoneNumber/taxCode/companyName), role mặc định `REVIEWER` luôn gán server-side (không tin client), company resolve theo taxCode → normalizedName → tạo mới với `verificationStatus=UNVERIFIED`, `verificationSource=MANUAL_REGISTRATION` (giá trị enum mới, không gọi TaxApiClient lúc đăng ký để giữ tốc độ)
- `AuthController.login()` viết lại: tra DB thật, `PasswordEncoder.matches()`, build JWT claims từ dữ liệu user thật thay vì hardcode
- Xác nhận tầng RBAC/JWT (`SecurityConfig.jwtAuthenticationConverter()`, mapping `roles` claim → `ROLE_` prefix) đã đúng và đầy đủ từ trước, không cần sửa

### 4.4 Audit trail cho Mô hình B (MLOps)
- Phát hiện & sửa: `reviewedBy` trong cả `DocumentApprovalService` lẫn `FieldAuditService` trước đó lấy từ DTO client gửi lên (có thể giả mạo), không lấy từ JWT — đã thêm `reviewed_by_role` vào schema + entity, tạo component dùng chung **`CurrentUserResolver`/`CurrentUser`** (package `com.company.bizocr.security`) đọc username/roles/companyId từ `SecurityContext`, thay thế toàn bộ chỗ đọc JWT claim rải rác trước đó (`DocumentController`, `DocumentApprovalController`, `FieldAuditService`)
- Phát hiện & sửa trùng lặp nghiêm trọng: `DocumentApprovalService.saveAuditLog()` (ghi gộp cả JSON vào 1 dòng, `field_name="extractedFields"`) bị xóa, thay bằng gọi `FieldAuditService.logFieldCorrections()` (diff từng field thật) — hợp nhất về 1 đường ghi audit duy nhất, vì Mô hình B cần `field_name` là tên field nghiệp vụ thật để lọc/đếm
- Xác nhận và xóa `DocumentReviewService.java` — một luồng "approve" thứ 2, độc lập, không còn được gọi từ controller nào (dead code)
- **Kết quả: điểm nghẽn dữ liệu tin cậy cho fine-tune đã được giải quyết triệt để** — login thật, audit trail theo từng field kèm người duyệt + role, chỉ 1 đường ghi audit duy nhất

---

## 5. Vấn đề tồn đọng quan trọng (phát hiện gần nhất)

### 5.1 Sai định nghĩa điều kiện gọi Ollama fallback
Chẩn đoán ban đầu (lỗi dấu tiếng Việt do OCR/threshold ảnh) **đã bị bác bỏ qua phân tích code thật**:
- `image_preprocessor.py`: bước threshold chỉ dùng để tìm contour/crop viền giấy, **không hề ảnh hưởng ảnh đưa vào OCR**
- `paddle_ocr_engine.py`: `lang="en"` trên PaddleOCR là đúng vì engine đó chỉ detect box, không đọc nội dung — việc đọc chữ có dấu do VietOCR đảm nhiệm

**Nguyên nhân thật đã xác định qua ví dụ cụ thể** (`Tân Cảng - Cái Mép` → `Cái Mếp` ở field đã trích xuất, trong khi `rawOcrText` gốc là `tan cang-cni mep` — không có mối liên hệ hợp lý nào giữa 2 chuỗi này bằng sửa lỗi OCR thông thường): hệ thống đang gọi Ollama **sai điều kiện** — dùng `confidence` (độ tin cậy đọc chữ) làm ngưỡng trigger, thay vì đúng ra phải là **độ hoàn chỉnh vật lý của tài liệu** (rách, tróc, cắt mất). Khi OCR chỉ đơn thuần đọc nhiễu (không phải tài liệu hư hại), hệ thống vẫn gọi Ollama, và LLM **bịa ra** một địa danh nghe hợp lý (`Cái Mép` có thật ở Việt Nam) thay vì báo không đọc được — đúng rủi ro Data Poisoning đã cảnh báo từ đầu.

### 5.2 Lộ trình khắc phục đã thống nhất (theo ưu tiên)

| # | Hạng mục | Trạng thái |
|---|---|---|
| P1 | Đổi trigger Ollama: `confidence` đơn lẻ → `completeness score` đa yếu tố (tỷ lệ field bắt buộc rỗng, tỷ lệ diện tích contour/ảnh gốc, avg confidence OCR) | Chưa bắt đầu — đang chờ `AiClassifierFallback.java`/`ClassifierService.java` để thiết kế công thức khớp code thật |
| P2 | Khóa cứng: output Ollama không bao giờ ghi thẳng vào field chính thức, luôn qua trạng thái gợi ý chờ xác nhận (`NEEDS_REVIEW`) | Chưa bắt đầu |
| P3 | Thêm tín hiệu `aspect_ratio` (đã có sẵn trong `image_preprocessor.py` metadata) vào `ClassifierService` để suy luận loại chứng từ theo kích thước | Chưa bắt đầu, ưu tiên thấp |
| P4 | Cache theo field định danh đã confirm (tax_code/phone/email...) để tránh gọi lại Ollama cho tài liệu đã duyệt — **không dùng so khớp ảnh** vì rủi ro false-positive cao trong hệ thống đa tenant | Hoãn lại, làm sau khi P1–P3 ổn định |
| — | Vòng lặp Fine-tune Mô hình B (LoRA training) | **Hoãn hẳn** — lần chạy hiện tại chỉ để kiểm tra hạ tầng (thông luồng, độ trễ, audit log ghi đúng), chưa dùng cải thiện model thật cho tới khi P1+P2 ổn định và có Golden Dataset sạch |

**Đánh giá lộ trình (thang 1–5, 10 tiêu chí): 46/50** — điểm bị trừ chủ yếu do `completeness score` cần dữ liệu thật để tune (rủi ro regression lên luồng approve khi sửa `AiClassifierFallback`/`ClassifierService`, thời gian tune không đoán trước chính xác được), không phải do lộ trình sai hướng.

---

## 6. Nợ kỹ thuật đã ghi nhận (chưa xử lý)

- `DocumentApprovalService.upsertCompany()` và các method tạo Company trong `CompanyResolverService` vẫn trùng lặp logic không đồng bộ — bản đầu không set `verificationStatus`/`verificationSource`/`matchConfidence`, bản sau có set. Chưa gộp để tránh đụng vào luồng approve đang chạy ổn định.
- Template matching: `TemplateMatcher` từng có ngưỡng gây misclassify namecard thành `ELECTRICITY_BILL` — cần tiếp tục tune.
- Production hardening còn dở: gỡ credentials hardcode, tách config theo profile (`application-local.yml`).

---

## 7. Định hướng thiết kế đã chốt (nguyên tắc xuyên suốt)

- **Mô hình cải tiến — Phễu quy tắc gia cố:** AI (Ollama) chỉ được gọi khi tầng rule-based không tự tin phân loại/trích xuất được — không gọi mặc định.
- **Mô hình B — Guardrailed MLOps Pipeline:** audit sạch (role đáng tin) → sanitize → fine-tune cách ly → eval trên golden set → auto-rollback → deploy có JSON Schema Guardrail. Nguyên tắc: JSON hợp lệ về cấu trúc không đồng nghĩa nội dung đúng — cần thêm lớp kiểm tra nội dung/độ hoàn chỉnh tài liệu trước khi tin dữ liệu để train.
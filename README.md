
Hệ thống phát hiện bệnh ở cây trồng bằng Trí tuệ Nhân tạo

Giới thiệu

Hệ thống **AI Plant Disease Detection** sử dụng mô hình Deep Learning (EfficientNet-B0) kết hợp với **Google Gemini AI** để:

1. **Nhận diện bệnh** trên lá cây trồng từ ảnh chụp với độ chính xác lên đến **99.30%**
2. **Tư vấn chi tiết** về triệu chứng, cách điều trị và phòng ngừa bệnh bằng tiếng Việt

Hệ thống hỗ trợ nhận diện **39 loại bệnh/trạng thái** trên **14 loại cây trồng** phổ biến, bao gồm: Táo, Việt quất, Anh đào, Ngô, Nho, Cam, Đào, Ớt chuông, Khoai tây, Mâm xôi, Đậu nành, Bí, Dâu tây và Cà chua.

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|--------|
| 🖼️ **Upload ảnh** | Tải ảnh lá cây từ thiết bị (JPG, PNG, JPEG) |
| 📷 **Chụp Camera** | Chụp ảnh trực tiếp từ camera thiết bị |
| 🔍 **Dự đoán Top-5** | Hiển thị 5 bệnh có xác suất cao nhất kèm độ tin cậy (%) |
| 🤖 **Tư vấn AI Gemini** | Phân tích chi tiết bằng Google Gemini với thông tin: tên bệnh, triệu chứng, cách điều trị, phòng ngừa |
| ⚠️ **Cảnh báo độ tin cậy** | Thông báo khi độ tin cậy dự đoán thấp (< 50%) |
| 🔄 **Multi-model Fallback** | Tự động chuyển đổi giữa nhiều model Gemini để đảm bảo tính ổn định |
| 🖥️ **Giao diện trực quan** | Web app thân thiện với Streamlit, hỗ trợ tiếng Việt |

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit Web App                 │
│                  (app_streamlit.py)                  │
├─────────────┬───────────────────┬───────────────────┤
│  📤 Upload  │   📷 Camera      │   📊 Hiển thị     │
│   ảnh lá    │   chụp trực tiếp │   kết quả         │
├─────────────┴───────────────────┴───────────────────┤
│                                                     │
│  ┌─────────────────┐    ┌─────────────────────────┐ │
│  │  EfficientNet-B0│    │   Google Gemini API      │ │
│  │  (PyTorch)      │───▶│   (Tư vấn bệnh)         │ │
│  │                 │    │                           │ │
│  │  • 39 classes   │    │  • Triệu chứng           │ │
│  │  • Top-5 predict│    │  • Cách điều trị          │ │
│  │  • Confidence % │    │  • Phòng ngừa             │ │
│  └────────┬────────┘    └─────────────────────────┘ │
│           │                                         │
│  ┌────────┴────────┐                                │
│  │ final_model.pth │                                │
│  │ (Pre-trained    │                                │
│  │  weights)       │                                │
│  └─────────────────┘                                │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 Mô hình & Huấn luyện

### Kiến trúc mô hình

- **Base model**: EfficientNet-B0 (pre-trained trên ImageNet)
- **Transfer Learning**: Thay thế lớp classifier cuối cùng (1280 → 39 classes)
- **Framework**: PyTorch
- **Training platform**: Google Colab (GPU T4)

### Chiến lược huấn luyện 2 giai đoạn

#### 🔹 Giai đoạn 1 — Feature Extraction
| Thông số | Giá trị |
|----------|---------|
| Epochs | 15 (early stopping tại epoch 8) |
| Learning Rate | 3×10⁻⁴ |
| Optimizer | AdamW (weight_decay=1×10⁻⁴) |
| Scheduler | CosineAnnealingLR (T_max=10) |
| Loss | CrossEntropyLoss |
| Augmentation | Mixup (α=0.4) |
| Patience | 3 epochs |
| **Best Accuracy** | **92.52%** |

#### 🔹 Giai đoạn 2 — Fine-tuning
| Thông số | Giá trị |
|----------|---------|
| Epochs | 5 |
| Learning Rate | 1×10⁻⁴ |
| Unfreeze | 3 lớp feature cuối cùng |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR (T_max=5) |
| Patience | 2 epochs |
| **Best Accuracy** | **99.30%** |

### Data Augmentation

```python
# Training transforms
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Mixup augmentation (α=0.4) - áp dụng trong giai đoạn 1
```

---

## 📊 Kết quả

### Hiệu suất tổng quan

| Metric | Giá trị |
|--------|---------|
| **Test Accuracy** | **99.30%** |
| **Train Accuracy** | 99.77% |
| **Top-3 Accuracy** | 99.97% |
| **Avg Confidence** | 0.9903 |
| **Min Confidence** | 0.3824 |
| **Max Confidence** | 1.0000 |
| **Precision (weighted)** | 0.99 |
| **Recall (weighted)** | 0.99 |
| **F1-Score (weighted)** | 0.99 |

### Chi tiết theo từng lớp (trích xuất)

| Cây trồng — Bệnh | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Apple — Apple scab | 1.00 | 0.96 | 0.98 | 131 |
| Apple — Black rot | 0.99 | 1.00 | 1.00 | 132 |
| Apple — Cedar apple rust | 1.00 | 1.00 | 1.00 | 58 |
| Corn — Gray leaf spot | 0.90 | 0.95 | 0.92 | 116 |
| Grape — Black rot | 1.00 | 1.00 | 1.00 | 223 |
| Orange — Greening | 1.00 | 1.00 | 1.00 | 1063 |
| Potato — Early blight | 1.00 | 1.00 | 1.00 | 215 |
| Tomato — Late blight | 0.99 | 1.00 | 0.99 | 407 |
| Tomato — Healthy | 0.99 | 0.99 | 0.99 | 332 |

> 📌 Tổng số mẫu test: **11,090** ảnh

---

## 🚀 Cài đặt & Chạy

### Yêu cầu hệ thống

- Python 3.10+
- CUDA-compatible GPU (khuyến nghị cho inference nhanh hơn, không bắt buộc)

### 1. Clone repository

```bash
git clone https://github.com/V-MinhQuan/AI_Detecting_Plants_Diseases.git
cd AI_Detecting_Plants_Diseases
```

### 2. Tạo môi trường ảo & cài đặt dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install torch torchvision streamlit google-generativeai Pillow
```

### 3. Cấu hình API Key

Thiết lập biến môi trường `GEMINI_API_KEY` cho tính năng tư vấn AI:

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_gemini_api_key_here"

# Windows (CMD)
set GEMINI_API_KEY=your_gemini_api_key_here

# Linux/macOS
export GEMINI_API_KEY="your_gemini_api_key_here"
```

> 🔑 Lấy API Key miễn phí tại: [Google AI Studio](https://aistudio.google.com/apikey)

### 4. Chạy ứng dụng

```bash
streamlit run app_streamlit.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

### 5. Huấn luyện lại mô hình (tùy chọn)

Mở file `code_model_nhom4.ipynb` trên Google Colab hoặc Jupyter Notebook để huấn luyện lại mô hình với dataset của bạn.

---

## 📁 Cấu trúc dự án

```
AI_Detecting_Plants_Diseases/
│
├── app_streamlit.py          # 🖥️ Ứng dụng web Streamlit (inference + tư vấn AI)
├── code_model_nhom4.ipynb    # 📓 Notebook huấn luyện mô hình (Google Colab)
├── final_model.pth           # 🧠 File trọng số mô hình đã huấn luyện (~16MB)
├── README.md                 # 📖 Tài liệu dự án
└── venv/                     # 🐍 Môi trường ảo Python
```

---

## 🗂️ Dataset

### Plant Leaf Diseases Dataset (Without Augmentation)

| Thông tin | Chi tiết |
|-----------|----------|
| **Tổng số ảnh** | ~55,448 |
| **Số lớp (classes)** | 39 |
| **Số loại cây** | 14 |
| **Train/Test split** | 80% / 20% |
| **Kích thước ảnh đầu vào** | 224 × 224 pixels |

### Danh sách 39 lớp bệnh

| # | Lớp | # | Lớp |
|---|-----|---|-----|
| 1 | Apple — Scab | 21 | Pepper — Healthy |
| 2 | Apple — Black Rot | 22 | Potato — Early Blight |
| 3 | Apple — Cedar Apple Rust | 23 | Potato — Late Blight |
| 4 | Apple — Healthy | 24 | Potato — Healthy |
| 5 | Background (without leaves) | 25 | Raspberry — Healthy |
| 6 | Blueberry — Healthy | 26 | Soybean — Healthy |
| 7 | Cherry — Powdery Mildew | 27 | Squash — Powdery Mildew |
| 8 | Cherry — Healthy | 28 | Strawberry — Leaf Scorch |
| 9 | Corn — Gray Leaf Spot | 29 | Strawberry — Healthy |
| 10 | Corn — Common Rust | 30 | Tomato — Bacterial Spot |
| 11 | Corn — Northern Leaf Blight | 31 | Tomato — Early Blight |
| 12 | Corn — Healthy | 32 | Tomato — Late Blight |
| 13 | Grape — Black Rot | 33 | Tomato — Leaf Mold |
| 14 | Grape — Esca (Black Measles) | 34 | Tomato — Septoria Leaf Spot |
| 15 | Grape — Leaf Blight | 35 | Tomato — Spider Mites |
| 16 | Grape — Healthy | 36 | Tomato — Target Spot |
| 17 | Orange — Huanglongbing (Greening) | 37 | Tomato — Yellow Leaf Curl Virus |
| 18 | Peach — Bacterial Spot | 38 | Tomato — Mosaic Virus |
| 19 | Peach — Healthy | 39 | Tomato — Healthy |
| 20 | Pepper — Bacterial Spot | | |

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mục đích |
|------------|----------|
| **PyTorch** | Framework Deep Learning, huấn luyện và inference mô hình |
| **EfficientNet-B0** | Kiến trúc CNN cho bài toán phân loại ảnh (Transfer Learning) |
| **Streamlit** | Xây dựng giao diện web tương tác |
| **Google Gemini API** | Tư vấn chi tiết về bệnh cây trồng bằng AI sinh tạo (Generative AI) |
| **Google Colab** | Nền tảng huấn luyện mô hình với GPU T4 miễn phí |
| **Pillow (PIL)** | Xử lý và tiền xử lý ảnh đầu vào |
| **scikit-learn** | Đánh giá mô hình (Classification Report, Confusion Matrix) |
| **Matplotlib / Seaborn** | Trực quan hóa kết quả huấn luyện và đánh giá |

---

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

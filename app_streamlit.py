import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import google.generativeai as genai
import os
import hashlib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Plant Disease Detection",
    layout="wide",
    page_icon="🌿"
)

# =========================
# GEMINI API KEY
# =========================
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Chưa có GEMINI_API_KEY")
    st.stop()

genai.configure(api_key=API_KEY)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CLASS NAMES
# =========================
class_names = [
    'Apple_scab','Apple_black_rot','Apple_rust','Apple_healthy',
    'Background','Blueberry_healthy','Cherry_powdery_mildew','Cherry_healthy',
    'Corn_gray_leaf_spot','Corn_rust','Corn_blight','Corn_healthy',
    'Grape_black_rot','Grape_esca','Grape_leaf_blight','Grape_healthy',
    'Orange_greening','Peach_bacterial_spot','Peach_healthy',
    'Pepper_bacterial_spot','Pepper_healthy',
    'Potato_early_blight','Potato_late_blight','Potato_healthy',
    'Raspberry_healthy','Soybean_healthy','Squash_mildew',
    'Strawberry_scorch','Strawberry_healthy',
    'Tomato_bacterial_spot','Tomato_early_blight','Tomato_late_blight',
    'Tomato_mold','Tomato_septoria','Tomato_spider_mite',
    'Tomato_target_spot','Tomato_yellow_leaf','Tomato_mosaic',
    'Tomato_healthy'
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 39)

    checkpoint = torch.load("final_model.pth", map_location=device)

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# GEMINI MULTI-MODEL FALLBACK
# =========================
def call_gemini(prompt):
    models_list = [
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-pro-latest",
        "models/gemini-3.1-flash-lite-preview",
        "models/gemini-3-pro-preview"
    ]

    for m in models_list:
        try:
            model = genai.GenerativeModel(m)
            res = model.generate_content(prompt)

            if res and res.text:
                return res.text

        except:
            continue

    return "❌ Tất cả Gemini models đều thất bại"

# =========================
# PREDICT FUNCTION
# =========================
def predict(image):
    # FIX lỗi 4 channels (RGBA)
    image = image.convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        top5_prob, top5_idx = torch.topk(prob, 5)

    return [
        (class_names[top5_idx[0][i].item()],
         top5_prob[0][i].item())
        for i in range(5)
    ]

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🌿 Hệ thống phát hiện bệnh ở cây trồng 🌿</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SESSION STATE
# =========================
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "results" not in st.session_state:
    st.session_state.results = None

if "answer" not in st.session_state:
    st.session_state.answer = None

# =========================
# INPUT MODE
# =========================
option = st.radio(
    "📥 Chọn nguồn ảnh",
    ["📤 Upload ảnh", "📷 Camera"]
)

uploaded = None

if option == "📤 Upload ảnh":
    uploaded = st.file_uploader(
        "Upload ảnh lá cây",
        type=["jpg", "png", "jpeg"]
    )

elif option == "📷 Camera":
    uploaded = st.camera_input("Chụp ảnh từ camera")

# =========================
# PROCESS IMAGE
# =========================
if uploaded:

    image = Image.open(uploaded).convert("RGB")

    file_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # RESET khi ảnh mới
    if st.session_state.file_hash != file_hash:
        st.session_state.file_hash = file_hash
        st.session_state.results = predict(image)
        st.session_state.answer = None

    results = st.session_state.results

    # =========================
    # UI
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    with col2:

        st.subheader("📊 Kết quả dự đoán")

        for i, (name, score) in enumerate(results):
            if i == 0:
                st.success(f"🥇 {name} ({score*100:.2f}%)")
            else:
                st.write(f"{i+1}. {name} ({score*100:.2f}%)")

        if results[0][1] < 0.5:
            st.warning("⚠️ Độ tin cậy thấp")

        # =========================
        # GEMINI
        # =========================
        if st.session_state.answer is None:

            diseases = ", ".join(
                [f"{n} ({p*100:.1f}%)" for n, p in results]
            )

            prompt = f"""
You are an agricultural expert.

Detected diseases:
{diseases}

Explain in Vietnamese:
- disease name (keep English)
- symptoms
- treatment
- prevention
"""

            with st.spinner("🤖 AI đang tư vấn..."):
                st.session_state.answer = call_gemini(prompt)

        st.subheader("🌱Kết quả tư vấn")
        st.success(st.session_state.answer)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>Made with group 4 Artificial Intelligence ELC3008_1</center>",
    unsafe_allow_html=True
)
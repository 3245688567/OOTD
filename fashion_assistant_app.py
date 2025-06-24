# Prepare a basic FAISS + YOLOv8 integration script template.
# This will be a streamlit app that:
# - Loads YOLOv8 to detect clothing items
# - Uses CLIP to embed cropped clothing items
# - Uses FAISS to find similar embeddings from a small mock dataset

streamlit_app_code = """
import streamlit as st
from PIL import Image
import torch
import clip
import faiss
import numpy as np
import os
from ultralytics import YOLO
from torchvision import transforms

# Initialize models and FAISS index
@st.cache_resource
def load_models():
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    yolo_model = YOLO('yolov8n.pt')  # You can replace this with a custom-trained fashion model
    return clip_model, clip_preprocess, yolo_model

clip_model, clip_preprocess, yolo_model = load_models()

# Mock dataset (precomputed embeddings + image names)
@st.cache_data
def load_mock_faiss_index():
    dim = 512
    index = faiss.IndexFlatL2(dim)
    # Mock 3 image embeddings
    mock_embeddings = np.random.rand(3, dim).astype("float32")
    index.add(mock_embeddings)
    image_ids = ["product_1.jpg", "product_2.jpg", "product_3.jpg"]
    return index, image_ids

index, image_ids = load_mock_faiss_index()

# Streamlit UI
st.title("👗 AI Fashion Matcher with YOLOv8 + FAISS")
uploaded_file = st.file_uploader("Upload a fashion photo", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Upload", use_container_width=True)

    # Run YOLOv8 detection
    results = yolo_model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    st.subheader("🔍 Detected Items & Matches")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cropped = image.crop((x1, y1, x2, y2))
        st.image(cropped, caption=f"Detected Item {i+1}")

        # Get CLIP embedding
        input_tensor = clip_preprocess(cropped).unsqueeze(0)
        with torch.no_grad():
            embedding = clip_model.encode_image(input_tensor).numpy().astype("float32")

        # Search FAISS
        D, I = index.search(embedding, k=3)
        st.markdown("**Top Matches:**")
        cols = st.columns(3)
        for j in range(3):
            with cols[j]:
                st.image(f"https://via.placeholder.com/150?text={image_ids[I[0][j]]}", caption=image_ids[I[0][j]])
"""

# Save the app script
streamlit_faiss_yolo_path = "/mnt/data/fashion_yolo_faiss_app.py"
with open(streamlit_faiss_yolo_path, "w") as f:
    f.write(streamlit_app_code.strip())

streamlit_faiss_yolo_path

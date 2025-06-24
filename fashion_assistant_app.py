# Updated Streamlit app with YOLOv8 detection, CLIP embedding, and FAISS similarity search

enhanced_streamlit_code = """
import streamlit as st
from PIL import Image
import numpy as np
import torch
import clip
import faiss
from torchvision import transforms
from ultralytics import YOLO

# Set up Streamlit page
st.set_page_config(page_title="AI Fashion Assistant", layout="centered")
st.title("👗 AI Fashion Assistant")
st.markdown("Upload a fashion photo, and we'll detect the items and find similar fashion products.")

# Load YOLOv8 and CLIP
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')  # Replace with a fashion-specific model if available
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    return yolo, clip_model, clip_preprocess

yolo_model, clip_model, clip_preprocess = load_models()

# Load mock FAISS index (in real case, use precomputed product vectors)
@st.cache_data
def load_faiss_index():
    dim = 512
    index = faiss.IndexFlatL2(dim)
    mock_embeddings = np.random.rand(3, dim).astype("float32")
    index.add(mock_embeddings)
    image_ids = ["product_dress.jpg", "product_jacket.jpg", "product_shoes.jpg"]
    return index, image_ids

faiss_index, image_ids = load_faiss_index()

# Upload and process image
uploaded_file = st.file_uploader("Upload an outfit image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("🧠 Detected Fashion Items:")
    results = yolo_model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cropped = image.crop((x1, y1, x2, y2))
        st.image(cropped, caption=f"Item {i+1}")

        # Get CLIP embedding
        clip_input = clip_preprocess(cropped).unsqueeze(0).to("cpu")
        with torch.no_grad():
            features = clip_model.encode_image(clip_input).numpy().astype("float32")

        # FAISS similarity search
        D, I = faiss_index.search(features, k=3)

        st.markdown("**Top Matches:**")
        cols = st.columns(3)
        for j in range(3):
            with cols[j]:
                st.image(f"https://via.placeholder.com/150?text={image_ids[I[0][j]]}", caption=image_ids[I[0][j]])
                
st.markdown("---")
st.caption("Powered by YOLOv8, CLIP & FAISS. Replace mock data with real fashion products for production.")
"""

# Save updated version of the Streamlit app
enhanced_app_path = "/mnt/data/fashion_assistant_app.py"
with open(enhanced_app_path, "w") as f:
    f.write(enhanced_streamlit_code.strip())

enhanced_app_path

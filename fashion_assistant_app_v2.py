import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from PIL import Image
import numpy as np
import torch
import clip
from torchvision import transforms
from ultralytics import YOLO
from duckduckgo_search import DDGS
import urllib.parse

# 📌 Set Streamlit UI
st.set_page_config(page_title="Shop the Look AI", layout="centered")
st.title("🛍️ Shop the Look: AI Fashion Assistant")
st.markdown("Upload a fashion photo — we detect outfits and find similar styles online.")

# 📦 Load Models (YOLOv8 + CLIP)
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')  # General model, replace with fashion-specific if you have
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    return yolo, clip_model, clip_preprocess

yolo_model, clip_model, clip_preprocess = load_models()

# 🔎 DuckDuckGo image search
def duckduckgo_search(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.images(keywords=query, max_results=max_results)
        return [r["image"] for r in results]

# 🧠 Encode outfit image using CLIP
def encode_clip_embedding(image: Image.Image):
    image_input = clip_preprocess(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        features = clip_model.encode_image(image_input)
    return features.cpu().numpy().astype("float32")

# 📤 Upload and process image
uploaded_file = st.file_uploader("Upload an outfit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("👕 Detected Fashion Items")
    results = yolo_model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) == 0:
        st.warning("No clothing items detected.")
    else:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cropped = image.crop((x1, y1, x2, y2))
            st.image(cropped, caption=f"Item {i+1}", width=200)

            # Encode with CLIP (ready for FAISS)
            embedding = encode_clip_embedding(cropped)

            # Use simple keyword to simulate text-query for now (since CLIP is not text2image in this step)
            default_search_query = f"fashion item clothing"

            st.markdown(f"🔍 Searching online for: **{default_search_query}**")
            image_urls = duckduckgo_search(default_search_query)

            cols = st.columns(len(image_urls))
            for j, url in enumerate(image_urls):
                with cols[j]:
                    st.image(url, caption=f"Match {j+1}", use_column_width=True)

st.markdown("---")
st.caption("Powered by YOLOv8, CLIP, and DuckDuckGo. FAISS catalog coming soon.")

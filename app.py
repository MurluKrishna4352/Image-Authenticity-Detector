import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Set to False once the .h5 files has been downloaded
SIMULATION_MODE = False

st.set_page_config(
    page_title="Image Authenticity Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model(model_choice):
    if SIMULATION_MODE:
        return None
    
    # Map selection to filename
    if model_choice == "ResNet50":
        file_path = "models/resnet50_model.h5" # change to local path 
    else:
        file_path = "models/efficientnetb0_model.h5" # change to local path 

    try:
        if os.path.exists(file_path):
            # Load Keras .h5 model
            model = tf.keras.models.load_model(file_path)
            return model
        else:
            st.error(f"⚠️ Model file not found: {file_path}")
            st.info("Please download the .h5 files")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. PREPROCESSING ---
def preprocess_image(image):
    # Resize
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Convert to array
    img_array = np.array(image)
    
    # Normalize pixel values [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

# --- 3. UI LAYOUT ---
st.title("🕵️ Image Authenticity Detector")
st.markdown("### Detect Real vs. AI-Generated Images")

# --- SIDEBAR ---
# 1. New About Section
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This system uses **Deep Learning** to classify images as either **Real** or **AI-Generated**.
    
    It analyzes pixel-level artifacts and noise patterns using models trained on 3,000 samples.
    """
)

st.sidebar.markdown("---")

# 2. Configuration Section
st.sidebar.header("⚙️ Configuration")
model_select = st.sidebar.selectbox(
    "Select Model Architecture", 
    ["ResNet50", "EfficientNetB0"],
    help="ResNet50 is more accurate, EfficientNet is faster."
)
st.sidebar.success(f"Loaded: **{model_select}**")

# --- MAIN CONTENT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("2. Prediction Results")
    
    if uploaded_file:
        
        # 1. Check Simulation Mode
        if SIMULATION_MODE:
             # Dummy logic
             import time, random
             with st.spinner("Simulating analysis..."):
                 time.sleep(1)
                 prob = random.random()
                 model_loaded = True

        # 2. Real Mode
        else:
            model = load_model(model_select) # This prints the error if missing
            
            if model:
                with st.spinner("Scanning for generative artifacts..."):
                    input_batch = preprocess_image(image)
                    prediction = model.predict(input_batch)
                    prob = float(prediction[0][0])
                    model_loaded = True
            else:
                model_loaded = False # STOP here if model is missing

        # 3. Display Results (ONLY if model loaded)
        if model_loaded:
            # Interpretation Logic
            if prob >= 0.5:
                label = "Real"
                confidence = prob
                is_ai = False
            else:
                label = "AI-Generated"
                confidence = 1.0 - prob
                is_ai = True
                
            st.markdown("---")
            
            if is_ai:
                st.error(f"🚨 **{label}**")
                st.write("The model detected statistical anomalies consistent with generative AI.")
            else:
                st.success(f"✅ **{label}**")
                st.write("The image noise profile matches authentic photography.")
                
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
            
            st.write("Probability Distribution:")
            st.progress(int(prob * 100))
            st.caption("0% = AI-Generated | 100% = Real Photo")
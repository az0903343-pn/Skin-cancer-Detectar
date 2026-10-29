import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================================
# 1. APPLICATION CONFIGURATION & WEB UI LAYOUT
# ==========================================================
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

st.title("🩺 Skin Cancer Detection using Deep Learning")
st.write("Provide a clear image of the skin lesion via Upload or Live Camera for an automated AI analysis.")

# Static classification label maps matching your HAM10000 data dictionary indices
CLASS_LABELS = {
    0: "Actinic keratoses (akiec) - Pre-cancerous",
    1: "Basal cell carcinoma (bcc) - Malignant Cancer",
    2: "Benign keratosis-like lesions (bkl) - Non-cancerous / Safe",
    3: "Dermatofibroma (df) - Non-cancerous / Safe",
    4: "Melanoma (mel) - High Risk Malignant Cancer",
    5: "Melanocytic nevi (nv) - Common Moles / Safe",
    6: "Vascular lesions (vasc) - Benign Blood Vessel Growth"
}

# ==========================================================
# 2. RUN COMPRESSED TFLITE MODEL INFERENCE ENGINE
# ==========================================================
@st.cache_resource
def load_tflite_model_safely():
    local_path = "skin_cancer_cnn_model_compressed.tflite"
    
    # Check if file exists and verify if it's a fake GitHub LFS pointer text file (usually < 1KB)
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        # If it's a small text file instead of actual binary model weights data
        if file_size < 2000:
            st.error("⚠️ GitHub uploaded file is an empty LFS Pointer instead of a raw binary asset.")
            st.info("💡 **Quick Solution:** Please delete the 'skin_cancer_cnn_model_compressed.tflite' file from GitHub, click 'Add File' -> 'Upload Files', and upload the 12.44 MB file again WITHOUT using git bash/command line tools.")
    
    # Initialize standard interpreter layers natively
    interpreter = tf.lite.Interpreter(model_path=local_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model_safely()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    st.error(f"System allocation breakdown. The current file content is empty or corrupted: {e}")
    st.info("🔄 Re-uploading the 12.44MB model file directly via GitHub web browser dashboard will solve this permanently.")
    model_loaded = False

# ==========================================================
# 3. DUAL INPUT SELECTION (UPLOAD OR LIVE CAMERA ACCESS)
# ==========================================================
input_mode = st.radio("Select Input Method:", ("Upload Image File", "Use Live Web Camera"))

raw_image_data = None

if input_mode == "Upload Image File":
    raw_image_data = st.file_uploader("Choose a skin lesion image file...", type=["jpg", "jpeg", "png"])
else:
    raw_image_data = st.camera_input("Position the skin lesion directly in front of the camera lens")

# ==========================================================
# 4. USER IMAGE UPLOAD PROCESSING PIPELINE
# ==========================================================
if raw_image_data is not None and model_loaded:
    image = Image.open(raw_image_data)
    st.image(image, caption="Target Skin Sample Image Selected.", use_container_width=True)
    
    st.write("⚙️ Processing image arrays and running AI diagnostic nodes...")
    
    # Preprocessing pipeline matching CNN graph standards
    img_rgb = image.convert("RGB")
    img_resized = img_rgb.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    input_batch = np.expand_dims(img_array, axis=0)
    
    # Execute Model Prediction
    interpreter.set_tensor(input_details['index'], input_batch)
    interpreter.invoke()
    
    prediction_scores = interpreter.get_tensor(output_details['index'])
    predicted_class_idx = int(np.argmax(prediction_scores))
    confidence_score = float(prediction_scores[predicted_class_idx] * 100)
    
    # ==========================================================
    # 5. DISPLAY REAL-TIME OUTPUT REPORT
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 System Diagnostic Assessment Result:")
    
    is_malignant = False
    if predicted_class_idx == 1:
        is_malignant = True
    if predicted_class_idx == 4:
        is_malignant = True

    if is_malignant:
        st.error(f"**Predicted Category:** {CLASS_LABELS[predicted_class_idx]}")
        st.warning("⚠️ Critical Notice: This signature exhibits malignant profiles. Please consult an expert dermatologist urgently.")
    else:
        st.success(f"**Predicted Category:** {CLASS_LABELS[predicted_class_idx]}")
        st.info("ℹ️ Information: This profile appears to be consistent with stable or benign growths.")
        
    st.metric(label="Model Predictive Confidence Score", value=f"{confidence_score:.2f}%")

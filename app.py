import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================================
# 1. APPLICATION CONFIGURATION & WEB UI LAYOUT
# ==========================================================
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

st.title("🩺 Skin Cancer Detection using Deep Learning")
st.write("Upload a clear image of the skin lesion to get an automated AI classification analysis.")

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
def load_tflite_model():
    # Targets the compressed model file in the same directory
    interpreter = tf.lite.Interpreter(model_path="skin_cancer_cnn_model_compressed.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model file. Please ensure 'skin_cancer_cnn_model_compressed.tflite' is uploaded: {e}")
    model_loaded = False

# ==========================================================
# 3. USER IMAGE UPLOAD PROCESSING PIPELINE
# ==========================================================
uploaded_file = st.file_uploader("Choose a skin lesion image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Open and display the user input image frame
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Skin Sample Image.", use_container_width=True)
    
    st.write("⚙️ Processing image arrays and running AI diagnostic nodes...")
    
    # Preprocessing: Convert to RGB, resize to match 128x128 input boundaries, normalize scale values
    img_rgb = image.convert("RGB")
    img_resized = img_rgb.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Expand array dimensions to match single-batch input specs
    input_batch = np.expand_dims(img_array, axis=0)
    
    # Invoke forward pass sequence inside the interpreter structure
    interpreter.set_tensor(input_details['index'], input_batch)
    interpreter.invoke()
    
    # Extract prediction probability output vector distribution arrays
    prediction_scores = interpreter.get_tensor(output_details['index'])
    predicted_class_idx = np.argmax(prediction_scores)
    confidence_score = prediction_scores[predicted_class_idx] * 100
    
    # ==========================================================
    # 4. DISPLAY REAL-TIME OUTPUT REPORT
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 System Diagnostic Assessment Result:")
    
    # Corrected syntax by explicitly initializing index verification boundaries
    if predicted_class_idx in:  # 1 for bcc, 4 for mel (malignant high risk targets)
        st.error(f"**Predicted Category:** {CLASS_LABELS[predicted_class_idx]}")
        st.warning("⚠️ Critical Notice: This signature exhibits malignant profiles. Please consult an expert dermatologist urgently.")
    else:
        st.success(f"**Predicted Category:** {CLASS_LABELS[predicted_class_idx]}")
        st.info("ℹ️ Information: This profile appears to be consistent with stable or benign growths.")
        
    st.metric(label="Model Predictive Confidence Score", value=f"{confidence_score:.2f}%")

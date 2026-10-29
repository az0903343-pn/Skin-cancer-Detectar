import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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
# 3. DUAL INPUT SELECTION (UPLOAD OR LIVE CAMERA ACCESS)
# ==========================================================
# Let the user choose how they want to provide the skin lesion image
input_mode = st.radio("Select Input Method:", ("Upload Image File", "Use Live Web Camera"))

raw_image_data = None

if input_mode == "Upload Image File":
    raw_image_data = st.file_uploader("Choose a skin lesion image file...", type=["jpg", "jpeg", "png"])
else:
    # Trigger the web browser native camera access interface feature directly
    raw_image_data = st.camera_input("Position the skin lesion directly in front of the camera lens")

# ==========================================================
# 4. USER IMAGE UPLOAD PROCESSING PIPELINE
# ==========================================================
if raw_image_data is not None and model_loaded:
    # Open the user input image frame safely using PIL wrapper
    image = Image.open(raw_image_data)
    st.image(image, caption="Target Skin Sample Image Selected.", use_container_width=True)
    
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
    predicted_class_idx = int(np.argmax(prediction_scores))
    confidence_score = float(prediction_scores[predicted_class_idx] * 100)
    
    # ==========================================================
    # 5. DISPLAY REAL-TIME OUTPUT REPORT
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 System Diagnostic Assessment Result:")
    
    # Check if malignancy profiles match bcc (1) or mel (4) categories
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

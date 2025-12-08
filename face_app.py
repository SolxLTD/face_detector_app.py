import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64

st.title("Interactive Viola-Jones Face Detector")
st.markdown("
Welcome to the Face Detection App! This tool uses the **Viola-Jones algorithm** (implemented via OpenCV's Haar Cascades) to detect frontal faces in an uploaded image.

# 📷 How to Use:
1.  **Upload an Image** (JPG/PNG) in the sidebar.
2.  **Adjust the Detection Parameters** (`Scale Factor` and `Min Neighbors`) in the sidebar to fine-tune the detection sensitivity.
3.  **Pick a Rectangle Color** for the bounding boxes.
4.  The processed image will appear below with detected faces highlighted.
5.  Click **'Download Processed Image'** to save the result.
---
")

CASCADE_FILE = "haarcascade_frontalface_default.xml" 

face_cascade = None
try:

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error(f"⚠️ Error: Could not load Haar Cascade from '{CASCADE_FILE}'. Please ensure the XML file is in the correct location.")
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred during cascade loading: {e}")


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    rgb = tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    # OpenCV uses BGR, so reverse the RGB tuple
    return rgb[::-1] 

st.sidebar.header("Detection Settings")

selected_color_hex = st.sidebar.color_picker(
    'Choose Rectangle Color', 
    '#FF0000', # Default to Red
    help="Select the color for the bounding boxes around detected faces."
)
bgr_color = hex_to_bgr(selected_color_hex)

scaleFactor = st.sidebar.slider(
    'Scale Factor',
    min_value=1.01,
    max_value=1.5,
    value=1.1,
    step=0.01,
    help="Determines the reduction rate of the image size at each detection scale. Lower values (e.g., 1.05) are more thorough but slower. Typical range: 1.1 - 1.3"
)

minNeighbors = st.sidebar.slider(
    'Minimum Neighbors',
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Specifies how many neighbors a candidate rectangle must have to be considered a face. Higher values reduce false positives but may miss real faces. Typical range: 3 - 6"
)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

processed_image_bytes = None

if uploaded_file is not None and face_cascade is not None and not face_cascade.empty():
    
    image_bytes = uploaded_file.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image_pil.convert('RGB'))
    
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    st.subheader("Original Image")
    st.image(image_pil, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

    st.subheader("Detected Faces Result")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(30, 30) # Minimum face size to consider
    )

    img_processed = img_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_processed, (x, y), (x + w, y + h), bgr_color, 2)
    
    img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    
    st.image(img_rgb, caption=f"Detected {len(faces)} faces using Scale Factor={scaleFactor} and Min Neighbors={minNeighbors}", use_column_width=True)
    
    if len(faces) == 0:
        st.warning("No faces detected with the current settings. Try adjusting the parameters or uploading a clearer image.")
    else:
        st.success(f"Detection Complete! Found {len(faces)} face(s).")
    
    img_save = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    img_save.save(buffer, format="PNG")
    processed_image_bytes = buffer.getvalue()
    
    st.session_state['processed_image'] = processed_image_bytes
    
# --- Download Button (2. Feature: Save Image) ---
if 'processed_image' in st.session_state and processed_image_bytes is not None:
    st.download_button(
        label="Download Processed Image (PNG)",
        data=st.session_state['processed_image'],
        file_name="detected_faces.png",
        mime="image/png",
        type="primary"
    )
else:
    st.info("Upload an image to start face detection and unlock the download option.")

st.sidebar.markdown("""
---
**Technical Note:**

This app relies on the Haar Cascade file (`haarcascade_frontalface_default.xml`) for the core detection logic.
""")

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io

# Set up the page
st.set_page_config(layout="wide", page_title="Image Segmentation App")

# Title and description
st.title("Image Segmentation App")
st.write("Upload an image to see the segmentation results")

# Create two columns for the image placeholders
col1, col2 = st.columns(2)

# Initialize session state for images
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'segmented_image' not in st.session_state:
    st.session_state.segmented_image = None

# Function to load and preprocess image
def preprocess_image(image):
    """Preprocess the image for the model"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Function to perform segmentation (placeholder - replace with your actual model)
def perform_segmentation(image_tensor):
    """
    Placeholder segmentation function.
    Replace this with your actual PyTorch model inference.
    """
    # This is a dummy segmentation - replace with your model
    # For demonstration, we'll create a simple threshold-based segmentation
    
    # Convert to numpy and process
    img_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
    img_np = img_np.astype(np.uint8)
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Convert back to PIL Image
    segmented_pil = Image.fromarray(segmented)
    
    return segmented_pil

# Function to handle image upload and processing
def process_uploaded_image(uploaded_file):
    """Process the uploaded image and perform segmentation"""
    try:
        # Read the image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Store original image
        st.session_state.original_image = image
        
        # Preprocess for model
        image_tensor = preprocess_image(image)
        
        # Perform segmentation
        with st.spinner('Performing segmentation...'):
            segmented_result = perform_segmentation(image_tensor)
        
        # Store segmented image
        st.session_state.segmented_image = segmented_result
        
        return True
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return False

# Display images in the first row
with col1:
    st.subheader("Original Image")
    if st.session_state.original_image:
        st.image(st.session_state.original_image, use_column_width=True)
    else:
        st.info("Upload an image to see the original")

with col2:
    st.subheader("Segmented Image")
    if st.session_state.segmented_image:
        st.image(st.session_state.segmented_image, use_column_width=True)
    else:
        st.info("Segmentation result will appear here")

# Second row - file uploader
st.markdown("---")
st.subheader("Upload Image")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Select an image file to segment"
)

if uploaded_file is not None:
    # Process the uploaded image
    success = process_uploaded_image(uploaded_file)
    
    if success:
        st.success("Segmentation completed successfully!")
        
        # Option to download the segmented image
        if st.session_state.segmented_image:
            # Convert segmented image to bytes for download
            buf = io.BytesIO()
            st.session_state.segmented_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Segmented Image",
                data=byte_im,
                file_name="segmented_image.png",
                mime="image/png"
            )

# Add some instructions
st.markdown("---")
st.markdown("""
### Instructions:
1. Click on 'Browse files' to upload an image
2. Supported formats: JPG, JPEG, PNG, BMP
3. The segmentation result will appear in the right panel
4. You can download the segmented image using the download button
""")

# Add a reset button
if st.button("Clear Images"):
    st.session_state.original_image = None
    st.session_state.segmented_image = None
    st.rerun()
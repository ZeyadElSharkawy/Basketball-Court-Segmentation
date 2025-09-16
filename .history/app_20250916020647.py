import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
import segmentation_models_pytorch as smp
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define these constants at the top of your file
NUM_CLASSES = 5  # Change this to your actual number of classes
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        A.Resize((512, 512)),
        A.Normalize(),
        ToTensorV2(),
    ])
    return transform(image).unsqueeze(0)

# Load your model once and cache it
@st.cache_resource
def load_segmentation_model(state_dict_path):
    """Load the trained segmentation model"""
    # Define model architecture
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    
    # Load state dictionary
    state_dict = torch.load(state_dict_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
    model.to(DEVICE)
    
    return model

# Function to perform segmentation with your SMP model
def perform_segmentation(image_tensor, model):
    """
    Perform segmentation using the trained SMP model
    """
    # Move input to device
    image_tensor = image_tensor.to(DEVICE)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        
        # Process output based on your model's output format
        if NUM_CLASSES == 1:
            # Binary segmentation - use sigmoid and threshold
            segmented_mask = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
            segmented_mask = (segmented_mask > 0.5).astype(np.uint8) * 255
        else:
            # Multi-class segmentation - take argmax
            segmented_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            # Normalize to 0-255 for visualization
            segmented_mask = (segmented_mask / (NUM_CLASSES - 1) * 255).astype(np.uint8)
    
    # Convert to PIL Image
    segmented_pil = Image.fromarray(segmented_mask)
    
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
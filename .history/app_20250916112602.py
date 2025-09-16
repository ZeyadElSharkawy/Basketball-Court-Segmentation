import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import segmentation_models_pytorch as smp

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define model architecture
NUM_CLASSES = 5

def create_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model

# Load trained model
@st.cache_resource
def load_model(model_path):
    model = create_model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        print("The application will use an untrained model.")
    
    model.to(DEVICE)
    model.eval()
    return model

# Preprocess image
def preprocess_image(img, img_size=(512, 512)):
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Convert to tensor
    tensor_img = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img, tensor_img

# Postprocess prediction
def postprocess_prediction(pred_mask, original_img_size):
    # Get the predicted class for each pixel
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_indices = pred_mask.squeeze().cpu().numpy()
    
    # Resize to original image size
    pred_indices_resized = cv2.resize(pred_indices, 
                                     (original_img_size[1], original_img_size[0]), 
                                     interpolation=cv2.INTER_NEAREST)
    return pred_indices_resized

# Draw lines on original image
def draw_lines_on_image(orig_img, pred_indices):
    # Define colors for each class (BGR format)
    class_colors_bgr = [
        (0, 0, 0),        # Class 0: Background (Black)
        (255, 0, 0),      # Class 1: Blue
        (0, 255, 0),      # Class 2: Green
        (0, 0, 255),      # Class 3: Red
        (255, 0, 255)     # Class 4: Magenta
    ]
    
    # Create result image and color mask
    result_img = orig_img.copy()
    color_mask_display = np.zeros((orig_img.shape[0], orig_img.shape[1], 3), dtype=np.uint8)
    
    # Draw contours for each class (skip background class 0)
    for i in range(1, NUM_CLASSES):
        class_mask = (pred_indices == i).astype(np.uint8)
        
        # Color the segmentation map
        color_mask_display[class_mask == 1] = class_colors_bgr[i]
        
        # Find contours and draw them
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, class_colors_bgr[i], 2)
    
    # Convert BGR to RGB for display
    color_mask_display_rgb = cv2.cvtColor(color_mask_display, cv2.COLOR_BGR2RGB)
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    return color_mask_display_rgb, result_img_rgb

def main():
    st.set_page_config(page_title="Court Line Segmentation", page_icon="🏀", layout="wide")
    
    st.title("🏀 Basketball Court Line Segmentation")
    st.markdown("""
    ### Instructions
    1. Upload a basketball court image.
    2. Click **Detect Court Lines** to run segmentation.
    3. View the segmentation map and overlay result side by side.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Model Configuration")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="D:\Personal Projects\Basketball Project\YOLO Results\U-net best model\\best_model_epoch89.pth"
    )

    uploaded_file = st.sidebar.file_uploader("Upload Court Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Small preview in sidebar only
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption="Uploaded Image (preview)", use_column_width=True)

        # Load model
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        if st.button("⚡ Detect Court Lines"):
            with st.spinner("Processing image..."):
                orig_img, tensor_img = preprocess_image(image)

                with torch.no_grad():
                    pred_mask_logits = model(tensor_img)

                pred_indices = postprocess_prediction(pred_mask_logits, orig_img.shape[:2])

                color_mask, result_img = draw_lines_on_image(orig_img, pred_indices)

                # Show results side by side (no duplicate original image)
                col1, col2 = st.columns(2)

                with col1:
                    st.image(color_mask, caption="Predicted Segmentation Map", use_column_width=True)

                with col2:
                    st.image(result_img, caption="Lines Overlay", use_column_width=True)

                # Legend
                st.subheader("Legend")
                legend_colors = [
                    ("Background", "Black"),
                    ("Half Court Line", "Blue"),
                    ("Out Lines", "Green"),
                    ("Free Throw Arc Line", "Red"),
                    ("Three Point Arc", "Magenta")
                ]
                for i, (line_type, color) in enumerate(legend_colors):
                    st.write(f"**{i}**: {line_type} ({color})")

if __name__ == "__main__":
    main()
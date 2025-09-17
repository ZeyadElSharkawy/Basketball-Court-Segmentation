import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
import segmentation_models_pytorch as smp
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define model architecture
NUM_CLASSES = 5

def create_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )

@st.cache_resource
def load_model_from_drive():
    model_path = "best_model_epoch89.pth"

    # ✅ Load credentials
    if os.path.exists("storage-472322-7676a7fe102c.json"):
        creds = service_account.Credentials.from_service_account_file(
            "storage-472322-7676a7fe102c.json"
        )
    elif "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    else:
        raise RuntimeError("❌ No Google Drive credentials found!")

    # ✅ Connect to Google Drive API
    drive_service = build("drive", "v3", credentials=creds)

    # 🔍 Your model’s file ID (ensure this is the ID of the file, not the folder)
    file_id = "1-La0m6zC-xbwi2MhayGihenDzg0Qew3w" # Make sure this is the correct file ID

    # ✅ Download model if missing
    if not os.path.exists(model_path):
        request = drive_service.files().get_media(fileId=file_id)
        
        # Use a 'with' statement to handle the file stream
        with io.FileIO(model_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download progress: {int(status.progress() * 100)}%")

    # ✅ Load PyTorch model
    # Now this will read a complete and properly saved file
    model = create_model() # Create an instance of the model first
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
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

    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5

    tensor_img = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img, tensor_img

# Postprocess prediction
def postprocess_prediction(pred_mask, original_img_size):
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_indices = pred_mask.squeeze().cpu().numpy()

    return cv2.resize(
        pred_indices,
        (original_img_size[1], original_img_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )

# Draw lines on original image
def draw_lines_on_image(orig_img, pred_indices):
    class_colors_bgr = [
        (0, 0, 0),        # Background
        (255, 0, 0),      # Half Court Line
        (0, 255, 0),      # Out Lines
        (0, 0, 255),      # Free Throw Arc Line
        (255, 0, 255),    # Three Point Arc
    ]

    result_img = orig_img.copy()
    color_mask_display = np.zeros((orig_img.shape[0], orig_img.shape[1], 3), dtype=np.uint8)

    for i in range(1, NUM_CLASSES):
        class_mask = (pred_indices == i).astype(np.uint8)
        color_mask_display[class_mask == 1] = class_colors_bgr[i]

        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, class_colors_bgr[i], 2)

    color_mask_display_rgb = cv2.cvtColor(color_mask_display, cv2.COLOR_BGR2RGB)
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    return color_mask_display_rgb, result_img_rgb

# Main app
def main():
    st.set_page_config(page_title="Court Line Segmentation", page_icon="🏀", layout="wide")

    # --- Sidebar for Future Work ---
    with st.sidebar:
        st.title("📝 Future Work & Project Roadmap")
        st.markdown("""
        This project demonstrates a proof-of-concept for basketball court line segmentation. The following enhancements are planned to improve its robustness and utility:
        """)
        
        st.subheader("Model & Training Improvements")
        st.markdown("""
        - **Upgrade Dataset:** Train the model using more diverse angles, lighting conditions, and court types to make it more robust and accurate.
        - **Advanced Augmentation:** Implement a more aggressive data augmentation pipeline (e.g., elastic transforms, cutout) to improve generalization.
        - **Explore Architectures:** Experiment with different model backbones (e.g., EfficientNet) or more advanced models like DeepLabV3+.
        - **Hyperparameter Tuning:** Systematically tune hyperparameters like learning rate, optimizer, and loss functions (e.g., Dice Loss, Focal Loss) for optimal performance.
        """)

        st.subheader("Application & Deployment")
        st.markdown("""
        - **Video Integration:** Upgrade the model to be used on videos, demonstrating how lines can be accurately drawn and classified during the broadcast of an actual basketball game.
        - **Real-Time Optimization:** Optimize the model for real-time inference using techniques like model quantization (e.g., with TensorRT) for smooth performance on live video.
        - **Post-Processing:** Add post-processing steps (e.g., morphological operations) to clean up predicted masks and create smoother, more consistent lines.
        """)

    # --- Main Application Area ---
    st.title("🏀 Basketball Court Line Segmentation")

    # --- NEW: Image Recommendations & Instructions ---
    st.markdown("### Instructions")
    st.info(
        """
        Upload a photo from one of the two angles below, as these are the angles the model was trained on. 
        The model was created to mainly work on the **Broadcaster Angle** for its potential use in real-world production.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        # IMPORTANT: Replace "broadcaster_angle.jpg" with the actual path to your image
        st.image("Block_by_Alexandre_Sarr_2025_01_08T00_35_28.jpg", caption="Broadcaster Angle")
    with col2:
        # IMPORTANT: Replace "floor_plan_angle.jpg" with the actual path to your image
        st.image("flickr_NBA_1183.jpg", caption="Floor Plan Angle")

    st.markdown("""
    1. Upload a basketball court image using the uploader below.
    2. Click **Detect Court Lines** to run segmentation.
    3. View the segmentation map and overlay results.
    """)
    # --- END of new section ---

    uploaded_file = st.file_uploader("Upload Court Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        model = load_model_from_drive()

        if st.button("⚡ Detect Court Lines"):
            with st.spinner("Processing image..."):
                orig_img, tensor_img = preprocess_image(image)

                with torch.no_grad():
                    pred_mask_logits = model(tensor_img)

                pred_indices = postprocess_prediction(pred_mask_logits, orig_img.shape[:2])
                color_mask, result_img = draw_lines_on_image(orig_img, pred_indices)

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(color_mask, caption="Predicted Segmentation Map", use_container_width=True)
                with res_col2:
                    st.image(result_img, caption="Lines Overlay", use_container_width=True)

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
import streamlit as st
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import cv2
import io,os
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, (H*W), C
        key = self.key(x).view(B, -1, H * W)  # B, C, (H*W)
        value = self.value(x).view(B, -1, H * W).permute(0, 2, 1)  # B, (H*W), C

        attention_scores = torch.matmul(query, key)  # B, (H*W), (H*W)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # B, (H*W), (H*W)
        attn_features = torch.matmul(attention_probs, value)  # B, (H*W), C
        attn_features = attn_features.permute(0, 2, 1).reshape(B, C, H, W)
        return attn_features

def resnet_model():
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    final_channels = model.fc.in_features  # 2048 for resnet50
    self_attention = SelfAttention(in_channels=final_channels, out_channels=final_channels)

    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(final_channels, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 5)
    )

    aux_head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(final_channels, 5)
    )

    model.aux_head = aux_head

    def forward_with_attention(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        attn_features = self_attention(x)
        pooled_features = F.adaptive_avg_pool2d(attn_features, (1, 1))
        flattened = pooled_features.view(pooled_features.size(0), -1)
        main_logits = model.fc(flattened)
        aux_logits = model.aux_head(x)

        return main_logits, aux_logits, attn_features

    model.forward = forward_with_attention

    return model

# Function to load the model
@st.cache_resource
def load_model():
    model = resnet_model()
    model.load_state_dict(torch.load('resnet.pth')['model_state_dict'])
    return model

import streamlit as st
import torch
import os
import tempfile
import zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, HiResCAM, XGradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
label_map = {0: 'EF-1', 1: 'EF-2', 2: 'EF-3', 3: 'EF-4', 4: 'EF-5'}

model= resnet_model()

saved_model = torch.load('resnet.pth')
model.load_state_dict(saved_model['model_state_dict'])

final_conv_layer = model.layer4[-1].conv3
target_layers = [final_conv_layer]

model.eval()
for param in model.parameters():
    param.requires_grad = True

# Available CAM methods
methods = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "layercam": LayerCAM
}

def load_and_preprocess_image(image_path):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (512, 512))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return rgb_img, input_tensor

def get_prediction(input_tensor):
    with torch.no_grad():
        outputs, _, _ = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
        return predicted.item(), confidence[predicted].item(), label_map[predicted.item()]
    
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        main_logits, _, _ = self.model(x)
        return main_logits

def create_zip_from_images(images_dict):
    """Create a zip file containing all CAM images."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, img_data in images_dict.items():
            zip_file.writestr(filename, img_data)
    zip_buffer.seek(0)  # Go to the beginning of the BytesIO buffer
    return zip_buffer

def visualize_cam(image_path, method_name="gradcam", target_category=None):
    rgb_img, input_tensor = load_and_preprocess_image(image_path)
    pred_idx, confidence, pred_label = get_prediction(input_tensor)

    targets = None
    if target_category is not None:
        targets = [ClassifierOutputTarget(target_category)]

    cam_algorithm = methods[method_name]
    wrapped_model = ModelWrapper(model)

    with cam_algorithm(model=wrapped_model, target_layers=target_layers) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Create visualization (CAM overlay)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Add label text overlay
        label_text = f"{pred_label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, label_text, (10, 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize only the output image to fixed width (e.g., 512px)
        fixed_width = 512
        h, w, _ = visualization.shape
        scale_ratio = fixed_width / w
        new_dim = (fixed_width, int(h * scale_ratio))
        visualization = cv2.resize(visualization, new_dim, interpolation=cv2.INTER_AREA)

    return pred_label, confidence, visualization

# Streamlit App
st.set_page_config(page_title="EF-Scale Classifier", layout="wide")
st.title("🌪️ Tornado Damage EF-Scale Classifier with Grad-CAM")

upload_mode = st.radio("Upload Mode:", ["Single Image", "Folder (ZIP)"])
method_choice = st.selectbox("Choose CAM Method", list(methods.keys()), index=0)

if upload_mode == "Single Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Get the CAM image
        label, conf, cam_image = visualize_cam(tmp_path, method_name=method_choice)

        # Display the resized CAM image
        st.image(cam_image, caption=f"Prediction: {label} ({conf:.2%})", use_column_width=False, width=600)

        # Convert the image to a downloadable format (PNG)
        cam_pil = Image.fromarray(cam_image)
        buf = io.BytesIO()
        cam_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Add a download button
        st.download_button(
            label="Download CAM Image",
            data=byte_im,
            file_name=f"{label}_cam.png",
            mime="image/png"
        )

else:
    uploaded_zip = st.file_uploader("Upload Folder as ZIP", type=["zip"])
    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            extract_dir = tempfile.mkdtemp()
            zip_ref.extractall(extract_dir)

        image_paths = [
            os.path.join(extract_dir, fname)
            for fname in os.listdir(extract_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        st.write(f"Processing {len(image_paths)} images using {method_choice}...")
        
        # Dictionary to hold image data for zipping
        images_for_zip = {}
        
        for image_path in image_paths:
            label, conf, cam_image = visualize_cam(image_path, method_name=method_choice)
            
            # Convert the CAM image to PNG format
            cam_pil = Image.fromarray(cam_image)
            buf = io.BytesIO()
            cam_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Save image data to dictionary with filenames
            filename = f"{label}_cam_{os.path.basename(image_path)}"
            images_for_zip[filename] = byte_im
            
            # Display CAM image for each image in the folder
            st.image(cam_image, caption=f"{os.path.basename(image_path)} → {label} ({conf:.2%})", use_container_width=False, width=600)

        # After all images are processed, create the zip file
        zip_buffer = create_zip_from_images(images_for_zip)

        # Provide download button for the zip file
        st.download_button(
            label="Download All CAM Images as ZIP",
            data=zip_buffer,
            file_name="cam_images.zip",
            mime="application/zip"
        )
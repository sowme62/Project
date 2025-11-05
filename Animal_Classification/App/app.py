# streamlit_resnet50_animal_app.py
# Single-file Streamlit app to run your ResNet50 animal classifier.
# Place your model file named exactly: "resnet50_animal_classifier (1).pth" in the same folder as this script

import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from PIL import Image
import streamlit as st

# --- Configuration ---
MODEL_PATH = "resnet50_animal_classifier.pth"  # change if your .pth has a different name or path
CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant",
    "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Unknown", "Zebra"
]
INPUT_SIZE = 224

st.set_page_config(page_title="ResNet50 Animal Classifier", layout="centered")
st.title("🐾 ResNet50 Animal Classifier")
st.write("Upload an image and the app will predict which animal class it likely belongs to.")

# --- Utilities ---
@st.cache_resource
def load_model(model_path=MODEL_PATH, device='cpu'):
    # Create a ResNet50 model with the correct final layer
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 16)

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        st.error(f"Failed to load model file '{model_path}': {e}")
        return None

    try:
        # ✅ Handle pure state_dict (OrderedDict)
        if isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state_dict = checkpoint

        # ✅ Handle dict with 'state_dict' key
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

        else:
            st.error("Unsupported checkpoint format. Expected a state_dict or checkpoint with 'state_dict' key.")
            return None

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Error while loading state_dict into model: {e}")
        return None


# Preprocessing follows standard ImageNet transforms used for ResNet
def preprocess_image(pil_img: Image.Image):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    return transform(pil_img).unsqueeze(0)  # add batch dim

# Prediction helper
@st.cache_resource
def get_model_and_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device=device)
    if model is None:
        return None, device
    model.to(device)
    return model, device

model, device = get_model_and_device()

if model is None:
    st.warning("Model not available. Please place your .pth file at the path shown above or update MODEL_PATH in the script.")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        st.image(pil_img, caption='Uploaded image', use_column_width=True)

        input_tensor = preprocess_image(pil_img).to(device)

        with st.spinner('Predicting...'):
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs[0], dim=0)
                topk = torch.topk(probs, k=3)

        top_probs = topk.values.cpu().numpy()
        top_idxs = topk.indices.cpu().numpy()

        st.subheader('Top predictions')
        for i, (idx, p) in enumerate(zip(top_idxs, top_probs)):
            st.write(f"{i+1}. {CLASS_NAMES[int(idx)]} — {p*100:.2f}%")

        # Optionally show a confidence gauge / simple message
        top1_conf = float(top_probs[0])
        if top1_conf < 0.4:
            st.info("Low confidence — the image might not belong to any of the trained classes or is ambiguous.")
        elif top1_conf < 0.75:
            st.success("Moderate confidence.")
        else:
            st.success("High confidence!")

    except Exception as e:
        st.error(f"Failed to process the uploaded image: {e}")

st.markdown("---")
st.write("Tips:\n- For best results upload a clear photo with the animal mostly centered and visible.\n- If you trained with a specific cropping/resolution, you may need to match those preprocessing steps in this file.")

# Footer: show model path so user knows which file to replace
st.caption(f"Model path used: {MODEL_PATH}")

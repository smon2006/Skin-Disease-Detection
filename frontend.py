import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

@st.cache_resource
def load_model():
    device = torch.device("cpu") # Safe fallback for free cloud servers
    model = models.efficientnet_b0(weights=None) 
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 22)
    )
    model.load_state_dict(torch.load("efficientnet_b0_skin_disease_74_9_acc.pth", map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Error loading model weights file: {e}")

class_names = [
    'Acne', 'Actinic Keratosis', 'Benign Tumors', 'Bullous Disease', 'Candidiasis', 
    'Drug Eruption', 'Eczema', 'Infestations / Bites', 'Lichen Planus', 'Lupus', 
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrheic Keratoses', 'Skin Cancer', 
    'Sun / Sunlight Damage', 'Tinea (Fungal Infection)', 'Unknown / Normal Skin', 'Vascular Tumors', 
    'Vasculitis', 'Vitiligo', 'Warts'
]

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

with st.sidebar:
    st.markdown("### 📋 About the System")
    st.write("This application uses a deep learning model (EfficientNet-B0) trained on 15,000+ clinical images to evaluate common skin conditions.")
    st.write("---")
    
    st.markdown("**Supported Conditions (22 Total):**")
    for condition in sorted(class_names):
        st.markdown(f"• {condition}")
        
    st.write("---")
    st.info("Developed for educational and portfolio purposes.")

st.title("🩺 Skin Disease Detection System")
st.markdown("### Powered by EfficientNet-B0 Computer Vision")
st.write("Upload a clinical image to get a real-time AI diagnosis.")

st.warning("**Medical Disclaimer:** This application is a machine learning research prototype intended for educational and portfolio purposes only. It is **not** a diagnostic tool. Always consult a certified dermatologist or qualified healthcare professional for actual medical advice and diagnoses.")

st.divider() 

col1, col2 = st.columns([1, 1]) 

with col1:
    st.markdown("### 1. Upload Clinical Image")
    uploaded_file = st.file_uploader("Upload a clear, well-lit image of the skin area (JPG/PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Processed Clinical Image', use_container_width=True)

with col2:
    st.markdown("### 2. Analysis Results")
    if uploaded_file is not None:
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.inference_mode():
                        logits = model(img_tensor)
                        probs = torch.softmax(logits, dim=1)[0]
                    
                    top3_prob, top3_idx = torch.topk(probs, 3)
                    
                    st.success("Analysis Complete!")
                    st.markdown("#### Top 3 Assessment Results:")
                    
                    for i in range(3):
                        condition = class_names[top3_idx[i].item()]
                        confidence = top3_prob[i].item()
                        
                        st.markdown(f"**{i + 1}. {condition}**")
                        st.progress(confidence, text=f"{confidence * 100:.2f}% Confidence")
                            
                except Exception as eval_error:
                    st.error(f"⚠️ Error: The system encountered an issue processing this image. {eval_error}")
    else:
        st.info("Upload an image on the left to see the analysis results here.")
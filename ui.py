import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import os
import torch.nn as nn
from torchvision import datasets
import plotly.express as px
import requests
from bs4 import BeautifulSoup

# Streamlit App Config
st.set_page_config(page_title="Plant Classifier", page_icon="🌿", layout="wide")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlantNet(nn.Module):
    def __init__(self, num_classes):
        super(PlantNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Trained Model
model_path = "plant_leaves_identify.pth"
if not os.path.exists(model_path):
    st.error("Model file not found! Please train and save 'plant_leaves_identify.pth'.")
    st.stop()

# Load dataset to get class names
data_dir = "plant_leaves_dataset"
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"))
class_names = train_dataset.classes
num_classes = len(class_names)

model = PlantNet(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Wikipedia Scraper Function
def get_plant_info(plant_name):
    search_url = f"https://en.wikipedia.org/wiki/{plant_name.replace(' ', '_')}"
    response = requests.get(search_url)
    if response.status_code != 200:
        return "No relevant information found.", None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract paragraph
    paragraphs = soup.find_all("p")
    plant_info = "No relevant information found."
    for p in paragraphs:
        if p.text.strip():
            plant_info = p.text.strip()
            break

    # Extract infobox image
    image = soup.find("table", {"class": "infobox"})
    img_url = None
    if image:
        img_tag = image.find("img")
        if img_tag:
            img_url = "https:" + img_tag["src"]

    return plant_info, img_url

# Streamlit UI
st.title("🌱 Plant Type Classifier")
st.write("Upload an image of a plant, and the AI model will predict its type.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess Image
    input_image = transform(image).unsqueeze(0).to(device)

    # Model Inference
    start_time = time.time()
    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    end_time = time.time()

    predicted_label = class_names[predicted_idx.item()]
    confidence = confidence.item() * 100
    inference_time = round((end_time - start_time) * 1000, 2)

    # Show Image & Prediction
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.success(f"### 🌱 Predicted: {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.warning(f"**Inference Time:** {inference_time} ms")

    # Metrics
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Class", predicted_label)
    col2.metric("Confidence Score", f"{confidence:.2f}%")
    col3.metric("Inference Time", f"{inference_time} ms")

    # Bar Chart - Top 5 Predictions
    top_k = 5
    top_probs, top_idxs = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy() * 100
    top_labels = [class_names[idx] for idx in top_idxs.cpu().numpy()]

    fig_bar = px.bar(
        x=top_labels, y=top_probs,
        title="Top-5 Prediction Confidence",
        labels={"x": "Plant Type", "y": "Confidence (%)"},
        color=top_probs, color_continuous_scale="greens"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Pie Chart - Confidence Distribution
    fig_pie = px.pie(
        names=top_labels, values=top_probs,
        title="Confidence Distribution",
        hole=0.3,
        color=top_labels,
        color_discrete_sequence=px.colors.sequential.Greens
    )
    st.plotly_chart(fig_pie)

    # Wikipedia Information
    st.subheader(f"📖 More About {predicted_label}")
    plant_info, plant_img = get_plant_info(predicted_label)

    if plant_img:
        st.image(plant_img, caption=f"{predicted_label} from Wikipedia", width=300)

    st.markdown(f"{plant_info}")

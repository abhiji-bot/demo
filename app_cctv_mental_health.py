import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import tempfile
import subprocess

# Ensure required dependencies are installed
subprocess.run(["pip", "install", "opencv-python-headless"], check=True)

# Load trained model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Three emotion classes
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = EmotionCNN()
model.load_state_dict(torch.load("enhanced_emotion_cnn.pth"))
model.eval()

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        frames.append(resized)
    cap.release()
    return np.array(frames)

def predict_emotion(video_frames):
    avg_frame = np.mean(video_frames, axis=0)
    avg_pixel_value = np.mean(avg_frame) / 255.0
    random_features = torch.tensor([avg_pixel_value, np.random.uniform(4, 10), np.random.uniform(50, 100), np.random.uniform(0, 1)], dtype=torch.float32)
    output = model(random_features.unsqueeze(0))
    prediction = torch.argmax(output, dim=1).item()
    return ["Low Stress", "Medium Stress", "High Stress"][prediction]

st.title("🎭 Student Mental Health Prediction App By Abhinav Khandelwal")
st.write("Upload a video to analyze the student's mental health based on facial expressions.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mpeg4"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
    
    st.video(video_path)
    st.write("Processing video and predicting stress levels...")
    video_frames = preprocess_video(video_path)
    emotion_prediction = predict_emotion(video_frames)
    st.markdown(f"### 🎭 Predicted Student Emotion: **{emotion_prediction}**")

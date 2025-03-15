import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define dataset class
class FacialExpressionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        label_mapping = {"Low": 0, "Medium": 1, "High": 2}  # Ensure zero-based indexing
        self.data["mental_health_risk"] = self.data["mental_health_risk"].map(label_mapping)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = torch.tensor(sample.iloc[:-1].values, dtype=torch.float32)
        label = torch.tensor(sample.iloc[-1], dtype=torch.long)
        return features, label

# Generate synthetic data with valid labels
def generate_synthetic_data(n=2000):  # Increased data size
    data = {
        "stress_level": np.random.randint(1, 11, n),
        "sleep_hours": np.random.uniform(4, 10, n),
        "attendance_rate": np.random.uniform(50, 100, n),
        "facial_emotion_score": np.random.uniform(0, 1, n),
        "mental_health_risk": np.random.choice(["Low", "Medium", "High"], n, p=[0.4, 0.4, 0.2])
    }
    return pd.DataFrame(data)

df = generate_synthetic_data()
df.to_csv("enhanced_synthetic_student_data.csv", index=False)

# Load dataset
dataset = FacialExpressionDataset(df)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Ensure output matches 3 classes
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train model
model = EmotionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=15):  # Increased epochs for better learning
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # Labels are zero-based now
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

train_model()

# Save trained model
torch.save(model.state_dict(), "enhanced_emotion_cnn.pth")
print("Model training complete and saved!")


# ============================================
# MODEL IMPROVEMENT STRATEGIES
# ============================================

# 1. Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# 2. Regularization (L2 Regularization)
from tensorflow.keras.regularizers import l2

l2_reg = l2(0.001)

# Example: Applying L2 Regularization to a Dense Layer
# model.add(Dense(128, activation='relu', kernel_regularizer=l2_reg))

# 3. Hyperparameter Optimization
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# 4. Ensemble Learning (Bagging Example)
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10)
# bagging_clf.fit(X_train, y_train)

# 5. Class Imbalance Handling (SMOTE)
from imblearn.over_sampling import SMOTE

smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 6. Transfer Learning (Using Pre-trained Model)
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freezing layers

# 7. Cross-Validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

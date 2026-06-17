# Device Classifier 📱🤖

An Android application that uses **TensorFlow Lite** to classify device images into 5 categories. The app allows users to select an image from the gallery and instantly receive predictions with confidence scores using an on-device machine learning model.

---

## 🚀 Features

- 📷 Select images from device gallery
- 🤖 On-device image classification using TensorFlow Lite
- ⚡ Real-time prediction with confidence scores
- 📊 Classifies images into 5 device categories
- 🔒 Fully offline (no cloud dependency)
- 🎨 Simple and user-friendly UI

---

## 🧠 Device Classes

The model predicts the following device types:

- D1-MC31  
- D2-WS50  
- D3-ET45  
- D4-WT64  
- D5-ZEC500  

---

## 🏗️ Tech Stack

- **Language:** Java  
- **Framework:** Android SDK  
- **ML Engine:** TensorFlow Lite (2.12.0)  
- **UI:** Material Design, ConstraintLayout  
- **Architecture:** On-device inference (offline ML)

---

## ⚙️ How It Works

1. User selects an image from the gallery
2. Image is resized to **224 × 224**
3. Pixel values are normalized (0–1 range)
4. TensorFlow Lite model processes the image
5. Highest confidence class is selected
6. Result is displayed with confidence score

---

## 📱 App Workflow
Image Selection → Preprocessing → TensorFlow Lite Model → Inference → Result Display

## 📲 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/bhasurapraneeth/Device-Classifier.git

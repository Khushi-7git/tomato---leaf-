🌾 Smart Agriculture System
-----
Crop Disease Detection & Crop Recommendation

This project combines Deep Learning and Machine Learning to help farmers and researchers:

🦠 Detect crop diseases from leaf images

🌱 Recommend the best crop based on soil and climate conditions

📌 Project Modules
--
1️⃣ Crop Disease Detection (Deep Learning)

Uses ResNet50 (Transfer Learning)

Classifies tomato leaf diseases

Trained on image dataset

Built using TensorFlow & Keras

2️⃣ Crop Recommendation System (Machine Learning)

Uses RandomForestClassifier

Predicts the most suitable crop based on soil nutrients & weather

Built using Scikit-learn

🧠 1. Crop Disease Detection Model
---
🔹 Description

A CNN-based image classification model using ResNet50 to detect tomato leaf diseases.

🔹 Key Features

Image augmentation

Transfer learning

Fine-tuning last layers

Softmax classification

🔹 Technologies Used

TensorFlow

Keras

ResNet50

ImageDataGenerator

🔹 Training Highlights

Image size: 224 × 224

Batch size: 32

Two-phase training:

Frozen base model

Fine-tuning last 30 layers

🔹 Model Architecture

ResNet50 (pretrained on ImageNet)

Global Average Pooling

Dense (128 units)

Dropout (0.3)

Output layer (Softmax)

🔹 Model Saving
model.save("tomato_disease_resnet50_finetuned.h5")

🔹 Example Prediction
Predicted Class: Tomato_Late_blight

🌱 2. Crop Recommendation System
---
🔹 Description

A machine learning model that recommends the best crop based on:

Nitrogen (N)

Phosphorus (P)

Potassium (K)

Temperature

Humidity

pH

Rainfall

🔹 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib & Seaborn

🔹 Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

🔹 Model Accuracy
Accuracy: ~99%

🔹 Confusion Matrix

Visualized using Seaborn heatmap

Shows excellent classification across crops like:

Rice

Maize

Coffee

Mango

Cotton

Banana

Apple

🔹 Saving the Model
joblib.dump(model, "crop_recommendation_model.pkl")

🔹 Example Predictions
sample_input = [90, 42, 43, 20.87, 82.0, 6.5, 202.93]
Recommended Crop: Rice

Multiple Test Cases
Soil & Climate Condition	Predicted Crop
High rainfall	Rice
Warm weather	Maize
Balanced nutrients	Coffee
Acidic soil	Mango

🛠 Installation
---
pip install -r requirement.txt

▶️ How to Run
Crop Recommendation
python crop_recommendation.py

Crop Disease Detection
python crop_disease_model.py


(Or run notebooks directly in Jupyter)

🌟 Future Enhancements
---

Streamlit web interface

Mobile app integration

More crop & disease classes

Real-time camera input

Cloud deployment

📜 License
---

This project is for educational and research purposes.

🙌 Acknowledgements
---

TensorFlow & Keras

Scikit-learn

Kaggle Datasets

Open-source community

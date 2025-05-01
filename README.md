# 🩺 Medical Diagnosis Prediction using Machine Learning

This project uses machine learning to predict the **outcome of a medical diagnosis** (Positive/Negative) based on patient symptoms and health profile. The model was trained on real-world-like data and is designed for quick clinical screening and educational use.

---

## 📌 Features

- Input: Symptoms like Fever, Cough, Fatigue, Difficulty Breathing, and patient profile (Age, Gender, Blood Pressure, Cholesterol).
- Output: Predicted medical outcome — **Positive** (likely illness) or **Negative**.
- Built using: Python, Scikit-learn, Streamlit.
- Interactive Web Interface (via Streamlit).

---

## 🧠 Machine Learning Model

- **Model**: Random Forest Classifier
- **Accuracy**: ~![image](https://github.com/user-attachments/assets/8f2c59a3-92a0-4695-ad26-43303a184f7d)

- **Data Preprocessing**: Label encoding for categorical features, train/test split
- **Target Variable**: `Outcome Variable` (Positive or Negative)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
https://github.com/balasai345/medical-diagnosis-predictor.git
cd medical-diagnosis-predictor
---

### 2. Install Requirements
pip install -r requirements.txt
or type manually
pip install pandas scikit-learn joblib streamlit matplotlib seaborn
### 3. Run the Web App
streamlit run app.py
## 📁 Project Structure
.
├── app.py                          # Streamlit web app
├── model-r.py                     # Model training script
├── medical_diagnosis_model.pkl    # Trained model
├── feature_encoders.pkl           # Encoders for input features
├── target_encoder.pkl             # Encoder for target
├── Disease_symptom_and_patient_profile_dataset.csv  # Input dataset
├── README.md                      # Project documentation
└── requirements.txt               # List of dependencies
## ScreenShots
![Screenshot 2025-05-01 153847](https://github.com/user-attachments/assets/299f2902-3527-421a-9c36-1209af82a330)
![Screenshot 2025-05-01 153928](https://github.com/user-attachments/assets/aa629f27-3be2-42ca-a001-ca6aa586f493)
📚 Acknowledgments
Dataset: Synthetic/curated medical dataset with symptoms and diagnosis.
Libraries: Scikit-learn, Streamlit, Pandas, Matplotlib, Seaborn.
⚠️ Disclaimer
This project is for educational purposes only. It is not a substitute for professional medical diagnosis.
## 🧑‍💻 Author
Your Name
www.linkedin.com/in/bala-sai-teja-jaddu-a6775028a | https://github.com/balasai345

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Drop target variable into y, and features into X
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

# Encode all categorical variables (both features and target)
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and encoders
joblib.dump(model, 'medical_diagnosis_model.pkl')
joblib.dump(label_encoders, 'feature_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

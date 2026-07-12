import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def generate_synthetic_medical_data():
    """Generates a synthetic medical dataset for training."""
    np.random.seed(42)
    num_samples = 1000

    # Features: Age, Blood_Pressure, Cholesterol, Glucose, Fever, Cough
    age = np.random.randint(18, 80, size=num_samples)
    bp = np.random.randint(90, 160, size=num_samples)
    cholesterol = np.random.randint(150, 280, size=num_samples)
    glucose = np.random.randint(70, 200, size=num_samples)
    fever = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    cough = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])

    # Targets logic (0: Healthy, 1: Diabetes, 2: Heart Disease, 3: Flu)
    target = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        if glucose[i] > 140:
            target[i] = 1  # Diabetes
        elif bp[i] > 130 and cholesterol[i] > 220:
            target[i] = 2  # Heart Disease
        elif fever[i] == 1 and cough[i] == 1:
            target[i] = 3  # Flu
        else:
            target[i] = 0  # Healthy

    df = pd.DataFrame(
        {
            "Age": age,
            "Blood_Pressure": bp,
            "Cholesterol": cholesterol,
            "Glucose": glucose,
            "Fever": fever,
            "Cough": cough,
            "Disease": target,
        }
    )
    return df


# --- Step 1: Load and Prepare Data ---
df = generate_synthetic_medical_data()
print("--- Dataset Sample ---")
print(df.head(), "\n")

# Separate features (X) and target variable (y)
X = df.drop("Disease", axis=1)
y = df["Disease"]

# --- Step 2: Split Data into Train and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 3: Train the Machine Learning Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 4: Evaluate the Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Healthy", "Diabetes", "Heart Disease", "Flu"],
    )
)

# --- Step 5: Predict Disease for a New Patient ---
# Custom Input: [Age, Blood_Pressure, Cholesterol, Glucose, Fever(0/1), Cough(0/1)]
new_patient_data = np.array([[55, 145, 240, 95, 0, 0]])

# Mapping prediction numbers back to names
disease_mapping = {0: "Healthy", 1: "Diabetes", 2: "Heart Disease", 3: "Flu"}

predicted_class = model.predict(new_patient_data)[0]
predicted_disease = disease_mapping[predicted_class]

print("--- Live Patient Prediction ---")
print(f"Patient Features: {new_patient_data[0]}")
print(f"Predicted Condition: {predicted_disease}")

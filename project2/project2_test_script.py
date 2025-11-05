# TEST SCRIPT FOR MODEL PREDICTION
#This script loads the trained Gradient Boosting Classifier (final_model.pkl) and runs predictions on a small sample dataset to verify model performance.
import pickle
import pandas as pd
# Load the trained and saved model back for testing
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the processed dataset
df = pd.read_csv("flat_file_labeled.csv")


# Take a small random sample (20 rows) for quick prediction test
sample = df.sample(20, random_state=42)
X_sample = sample.drop(columns=['target', 'risk_score'])
y_true = sample['target']

# Predict on sample data
y_pred = model.predict(X_sample)

# Display predicted vs actual risk classes for quick validation
print("True vs Predicted Risk Classes:")
result = pd.DataFrame({
    'true_target': y_true.values,
    'predicted_risk_class': y_pred
})
print(result)


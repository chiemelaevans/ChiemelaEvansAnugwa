# TEST SCRIPT FOR MODEL PREDICTION
#This script loads the trained Gradient Boosting Classifier (final_model.pkl) and evaluates it using the saved hold-out test set (test_data.pkl).
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score

# Load the trained and saved model back for testing
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved test data
with open("test_data.pkl", "rb") as f:
    X_test, y_test = pickle.load(f)


# Predict on test set
y_pred = model.predict(X_test)


# === DISPLAY FIRST 30 PREDICTIONS ===
result = pd.DataFrame({
    'true_target': y_test.values[:30],
    'predicted_risk_class': y_pred[:30]
})
print("=== True vs Predicted (First 30 Samples) ===")
print(result)

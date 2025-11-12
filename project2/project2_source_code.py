# ===  LIBRARIES ===
import os #operating system bridge
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import matplotlib.pyplot as plt


# === DATA PROCESSING ===
folder = r"C:\Users\OLUCHI\OneDrive\Desktop\project2\project2_raw_data"  # Folder containing all monthly .txt files
files = [f for f in os.listdir(folder) if f.endswith('.txt')]  # List of all .txt files in folder
all_data = []  # To hold data from all files

# Loop through each file and extract risk score + features
for file in files:
    with open(os.path.join(folder, file)) as f:
        for line in f:
            parts = line.strip().split()  # Split each row by space
            risk = float(parts[0])  # First column = risk score
            # Extract feature_number:feature_value pairs into dictionary
            feats = {f'feature_{int(p.split(":")[0])}': float(p.split(":")[1]) for p in parts[1:] if ':' in p}
            feats['risk_score'] = risk  # Add risk score
            feats['target'] = int(risk >= 0.3)  # Binary target (1 = high risk, 0 = low risk)
            all_data.append(feats)

# Convert all dictionaries to a single DataFrame, filling missing features with 0
df = pd.DataFrame(all_data).fillna(0)

# === DATA CLEANING ===
# Ensure consistent feature columns (feature_0 to feature_482 + risk_score + target)
cols = [f'feature_{i}' for i in range(483)] + ['risk_score', 'target']
df = df.reindex(columns=cols, fill_value=0)

# Save initial flat file
df.to_csv("flat_file.csv", index=False)
print(df.head())
df.info()

# Load feature mapping (maps feature_number to feature_name)
mapping = pd.read_csv(r"C:\Users\OLUCHI\OneDrive\Desktop\project2\feature_name_to_number_mapping.csv")

# Check for duplicates
if mapping['feature_name'].duplicated().any():
    print("Duplicate feature names found. Making them unique...")
    
    # Create unique names by appending feature_number
    mapping['feature_name'] = mapping.apply(
        lambda row: f"{row['feature_name']}_{int(row['feature_number'])}",
        axis=1
    )

# Create dictionary to rename columns using mapping file
new_dict = {f"feature_{int(row.feature_number)}": row.feature_name 
            for _, row in mapping.iterrows()}

# Rename columns to use descriptive feature names
df.rename(columns=new_dict, inplace=True)

# Save labeled dataset for modeling
df.to_csv("flat_file_labeled.csv", index=False)
print(df.head())
df.info()

# === PREDICTIVE MODELLING ===
# === Train-Test Split ===
# Split dataset into training (80%) and hold-out test (20%) sets
X = df.drop(columns=['target', 'risk_score'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state = 42
)
# Save the test set for later evaluation
with open("test_data.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

# === Model Training and Validation===
model = GradientBoostingClassifier(random_state = 42) # Initialize Gradient Boosted Trees model

# Define stratified cross-validation with fixed random seed
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)

# Perform cross-validation to evaluate precision and recall
cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=['precision', 'recall'],
    n_jobs=-1
)

# === MODEL OUTPUTS ===
prec, recall = cv_results['test_precision'], cv_results['test_recall']
print(f"Cross-Validation Precision: {prec.mean():.3f}")
print(f"Cross-Validation Recall:    {recall.mean():.3f}")

# Retrain final model on full training data
model.fit(X_train, y_train)


# === FEATURE IMPORTANCE VISUALIZATION ===
# Calculate and sort feature importances
importances = pd.Series(model.feature_importances_, index=X_train.columns)
top_20_features = importances.sort_values(ascending=False).head(20)

# Plot top 20 most important features
plt.figure(figsize=(8,5))
top_20_features.plot(kind='barh')
plt.gca().invert_yaxis()  # Highest importance on top
plt.title("Top 20 Most Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# === MODEL DEPLOYMENT ===
# Save final trained model as pickle file
with open("final_model.pkl", "wb") as f:
    pickle.dump(model, f)




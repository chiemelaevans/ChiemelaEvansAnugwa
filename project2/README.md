Overview

This project builds and deploys a Gradient Boosting Classifier to predict whether a software executable is high risk or low risk, based on sparse feature data extracted from 45 monthly .txt files. The model processes raw LIBSVM-style input data, trains a machine learning model, and packages it into a .pkl file for easy reuse and deployment.


Packaged Model Instructions

The final model is saved as final_model.pkl. Input data must be a pandas DataFrame with exactly 483 feature columns corresponding to Windows API/system call counts (e.g., NtCreateFile, RegOpenKey), named as listed in feature_name_to_number_mapping.csv. All features must be numeric (missing/absent calls should be encoded as 0). Do not include target or risk_score columns in input data. The model outputs a binary prediction: 0 = low-risk executable (risk score < 30%), 1 = high-risk executable (risk score ≥ 30%). The test script (saved as project2_test_script_2.py) demonstrates this binary prediction by loading the pickled model (final_model.pkl) and  using ten manually created sample dataset, and printing both true and predicted risk classes side by side for validation.

Requirements

To use the packaged model, first ensure you have Python 3.8+ with pandas, scikit-learn, and numpy installed. Python’s pickle module is used to save the trained model. Note that matplotlib is needed on the other hand to produce model summary plot. If these libraries are not already installed, install them using pip:
pip install pandas numpy scikit-learn matplotlib


Reproducing/ Retraining the model

-	Data Preparation
  
Place all monthly .txt data files in the project2_raw_data folder.
Each file should follow this LIBSVM-style format: 
0.57 12:1 19:1 22:4 24:10 34:1
Where:
First value = risk score (0–1), Subsequent values = feature_number:feature_value. Also, include the feature mapping file: feature_name_to_number_mapping.csv

-	Run Full Pipeline
  
Run the full training script (project2_source_code_2.py). This will process and clean the raw text files, merge into a labeled flat file (flat_file_labeled.csv), train and cross-validate a Gradient Boosting Classifier, generate a feature importance plot and save the final trained model as final_model.pkl.

-	Model Output Files
  
After training completes, the following files are generated in the working directory: flat_file.csv (flattened numeric data), flat_file_labeled.csv (feature-labelled data) and final_model.pkl (packaged trained model).

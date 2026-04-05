# Coronary Artery Disease (CAD) prediction

The purpose of this project is to create a machine learning model that can predict whether a patient has Coronary Artery Disease (CAD) as well as the likelihood of them having it represented as percentage.

## Features

- Data Preparation
- Data Analysis
- Feature Selection
- Modeling - kNN, Decision Tree, Random Forest, Logistic Regression
- Model Performance Comparison - accuracy, ROC-AUC Curves

## Dataset

- UCI Heart Disease Dataset (Cleveland Cohort) - 13 features, 303 instances
- Mendeley Data Cardiovascular Disease Dataset - 12 features, 1000 instances

## Data Preparation
- handled missing values 
- converted to numeric values 
- mapped categorical values to 0-based 
- gave the target variable binary values (0/1)
- normalized the second dataset to follow the naming conventions established in the first one
- dropped uncommon columns and added region column for clarity
- merged the two datasets into one (13 features, 1303 instances)
- scaled numeric features and one-hot encoded categorical ones  before modeling
  
## Data Analysis
- box plot (for numeric features)
- bar charts (for categorical features)
- correlation matrix
- scatter matrix (for numeric features)

## Feature Selection
- Cleveland dataset - age, thalach, oldpeak, exang, sex, cp, slope, ca, thal
- Merged dataset - trestbps, oldpeak, sex , slope, ca, fbs, restegc

## Modelling
- Models for the merged dataset showed higher accuracy overall
- Random Forest Merged Dataset - highest accuracy (93%)
- Decision Tree Cleveland Dataset - lowest accuracy (87%)


## ROC - AUC Curves 
- Models for the merged dataset showed high AUC overall
- Random Forest and Decision Tree Merged Dataset - highest AUC (0.98)
- Decision Tree Cleveland Dataset - lowest AUC (0.87)

## Conclusion 
- All models demonstrated strong classification performance 
- Models using the merged dataset demonstrated better performance - more diverse and representative examples 
- Model using the Cleveland dataset demonstrated high accuracy but lower than the prior - smaller datasets  limit the model’s ability to capture complex relationships
- Random Forest trained on the merged dataset stands out as the most accurate and generalizable model

## Explainable AI

- Core Ethical Challenges for AI implementation in Medicine
- SHAP Values

## Technologies Used

- Pyhton
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Sklearn
- Flask

## Self - Reflection
- Learned the full ML workflow from data prep to evaluation
- Understood the impact of clean, well-prepared data
- Faced challenges interpreting medical data responsibly
- Improved data visualization skills
- Practiced model comparison
- Learned the importance of explainability in healthcare models
- Realized ML supports but does not replace human expertise

## Demo
https://cad-ml-model-pz93.vercel.app/




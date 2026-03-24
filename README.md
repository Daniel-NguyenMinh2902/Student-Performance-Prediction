# Student-Performance-Prediction
Overview
- A machine learning project to predict students at risk of academic failure using socio-academic data.
> Goal: Build an early warning system to help identify students who need support before failure occurs.

Problem Statement
- Academic failure can have long-term educational and social consequences. Early identification of at-risk students allows institutions to intervene proactively.
- This project aims to build a predictive model that classifies students as "at risk" or "not at risk" ("safe"), based on academic, behavioral, and socio-economic factors.

Objectives
- Predict student failure risk (by building a binary classification model)
- Explore and analyze educational data
- Identify the most important factors influencing academic performance
- Compare multiple machine learning models
- Evaluate performance using appropriate metrics
- Provide interpretable insights for educational decision-making

Project Structure
Student-Performance-Prediction/
- data-student/        # Dataset files
- notebooks/           # Jupyter notebooks 
- reports/             # Results, reports, notes, analysis 
- src/                 # Images, diagrams 

- README.md
- .gitignore


Dataset
- Source: Student Performance - UCI Machine Learning Repository 
- Subjects: Mathematics and Portuguese ( Mathematics used in this project ) 
- Description:
This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful (see paper source for more details).
cre: https://archive.ics.uci.edu/dataset/320/student+performance

- Features include:
+) Academic: G1, G2, failures,...
+) Behavioral: absences, studytime, goout,...
+) Socio-economic: parental education (Medu, Fedu), family support (famsup),...
- Target: 
+) risk = 0 => student safe (G3 >= 10)
+) risk = 1 => student at risk (G3 < 10) 

Project Workflow
Repository setup : Done
1. Data exploration (EDA) : Done
2. Data Preprocessing : Done
- One-hot encoding
- Feature scaling
3. Model Training : Done
4. Model Evaluation & comparison : Done
5. Feature Importance & Interpretation : Done

Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature distribution analysis
- Correlation analysis
- Handling missing values: No missing values 

Models Used for Training 
- Logistic Regression (baseline)
- Decision Tree
- Random Forest (main model for feature importance)
- K-Nearest Neighbors KNN

Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

Installation
git clone https://github.com/Daniel-NguyenMinh2902/Student-Performance-Prediction.git
cd Student-Performance-Prediction

Usage
Run notebook in Terminal: jupyter notebook

Future Improvements
- Remove G1/G2 for earlier prediction
- Hyperparameter tuning
- Feature selection optimization
- Cross-validation
- Model deployment (API or web app)
- Real-time prediction system

Author
Duc Minh NGUYEN (Daniel Nguyen Minh)
- Computer Science Student (Third year CS bachelor student at Sorbonne University)
- Interested in Data Science & Machine Learning

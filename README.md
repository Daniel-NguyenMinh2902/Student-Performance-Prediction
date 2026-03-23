# Student-Performance-Prediction
Overview
- This project aims to predict student academic performance using machine learning techniques.
- The goal is to identify students at risk early and support data-driven decision-making in education.

Objectives
- Build a binary classification model (e.g., at risk vs not at risk)
- Explore and analyze educational data
- Compare multiple machine learning models
- Evaluate performance using appropriate metrics

Project Structure
Student-Performance-Prediction/
- data-student/        # Raw and processed datasets
- notebooks/           # Jupyter notebooks (EDA, experiments)
- src/                 # Figures, outputs (models, preprocessing)
- reports/             # Results, reports, notes

- README.md
- .gitignore


Dataset
- Source: Student Performance - UCI Machine Learning Repository 
- Description:
This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful (see paper source for more details).
cre: https://archive.ics.uci.edu/dataset/320/student+performance

Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature distribution analysis
- Correlation analysis
- Handling missing values: No missing values 

Models Used
- Logistic Regression (baseline)
- Decision Tree
- Random Forest (main model for feature importance)
- K-Nearest Neighbors (KNN)

Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

Installation
git clone https://github.com/Daniel-NguyenMinh2902/Student-Performance-Prediction.git
cd Student-Performance-Prediction

Usage
Run notebooks: jupyter notebook

Project Status
(In progress)
Repository setup : Done
Data exploration : Done
Feature engineering : Done
Model training : Done
Evaluation & comparison : Done
Interpretation : Done
Presentation : In progress

Future Improvements
- Hyperparameter tuning
- Feature selection optimization
- Cross-validation
- Model deployment (API or web app)
- Real-time prediction system

Author
Duc Minh NGUYEN (Daniel Nguyen Minh)
- Computer Science Student (Third year CS bachelor student at Sorbonne University)
- Interested in Data Science & Machine Learning

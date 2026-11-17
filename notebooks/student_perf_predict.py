#!/usr/bin/env python
# coding: utf-8

# ###
# ### --- STUDENT PERFORMANCE PREDICTION --- ###
# 
# Early predictive classification model that estimates the probability of student academic failure based on socio-academic factors
# (Early prediction of student academic failure risk)
# 
# ###

# In[1]:


import sys
print(sys.executable)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib as mtpl
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

print("\tPython ",sys.version)
print("\tpandas: ",pd.__version__)
print("\tnumpy: ",np.__version__)
print("\tmatplotlib: ",mtpl.__version__)


# ### Part 1 & 2: Find Dataset & Exploratory Data Analysis EDA ###

# In[3]:


df_mat = pd.read_csv("../data-student/student-mat.csv", sep=";")
df_por = pd.read_csv("../data-student/student-por.csv", sep=";")


# In[4]:


# first 5 lines of dataset 
df_mat.head()


# In[5]:


df_mat.shape


# In[6]:


df_mat.info()


# In[7]:


df_mat.describe()


# In[8]:


# target variable - detect risk 
df_mat["risk"] = (df_mat["G3"] < 10).astype(int)
df_mat["risk"].value_counts()
# G3 >= 10 => Success(0) - Risk = 0 False
# G3 < 10 => Fail(1) - Risk = 1 True 


# In[9]:


# percentage
df_mat["risk"].value_counts(normalize=True)


# In[10]:


sns.countplot(x="risk", data=df_mat)
plt.title("Distribution of academic Risk")
plt.savefig("../src/distribution_academic_risk.png")
plt.show()


# In[11]:


# Potentially strong predictors: studytime, (previous) failures, absences, G1, G2
# show which variables correlate risk classes

# studytime
sns.histplot(df_mat["studytime"], kde=True)
plt.savefig("../src/studytime.png")


# In[12]:


sns.boxplot(x="risk", y="studytime", data=df_mat)
plt.savefig("../src/studytime_box.png")


# In[13]:


# (previous) failures
sns.boxplot(x="risk", y="failures", data=df_mat)
plt.savefig("../src/failures_box.png")


# In[14]:


# absences
sns.boxplot(x="risk", y="absences", data=df_mat)
plt.savefig("../src/absences_box.png")


# In[15]:


# G1
sns.boxplot(x="risk", y="G1", data=df_mat)
plt.savefig("../src/G1_box.png")


# In[16]:


# G2
sns.boxplot(x="risk", y="G2", data=df_mat)
plt.savefig("../src/G2_box.png")


# In[17]:


# check correlation between (studytime, (previous) failures, absences, G1, G2) vs G3
plt.figure(figsize=(10,8))
sns.heatmap(df_mat.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")

plt.savefig("../src/correlation_matrix.png")
plt.show()


# In[18]:


# categorical variables (optional)
# do students want higher education fail less?
#sns.countplot(x="higher", hue="risk", data=df_mat)


# In[19]:


# do students with famsup fail less?
#sns.countplot(x="famsup", hue="risk", data=df_mat)


# In[20]:


# compare means by risk (optional)
# average differences between risk groups
df_mat.groupby("risk").mean(numeric_only=True)


# In[21]:


df_mat.corr(numeric_only=True)["G3"].sort_values(ascending=False)


# ### Part 3: Data Preprocessing ###

# In[23]:


# Define features and Target
# features : everything except "G3" and "risk"
X = df_mat.drop(columns = ["G3", "risk"])

# Target : "risk"
Y = df_mat["risk"] # 0 or 1


# In[24]:


# categorical variables :
X.select_dtypes(include="object").columns

# ML cannot read text categories
# so must convert them into numbers


# In[25]:


# Encode categorical variables :
# One-hot Encoding :
X_encoded = pd.get_dummies(X, dtype=int) # convert categorie 'object' into 'int64'
X_encoded


# ### Part 4: Model Training ###

# In[27]:


# train / test split : using 80% of data to train the model and 20% to evaluate how well it learns 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42, shuffle=True, stratify=Y)

print('X_train : ')
print(X_train.head())
print('')
print('X_test : ')
print(X_test.head())
print('')
print('y_train : ')
print(y_train.head())
print('')
print('y_test : ')
print(y_test.head())


# In[28]:


# Feature scaling : normalize the range of variables (features)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit scaler on X_train
X_test_scaled = scaler.transform(X_test) # test set remains unseen, if not the information'd be leaked


# ### Part 5: Model Evaluation ###

# In[30]:


# LOGISTIC REGRESSION model (baseline) : for classification problems, predicts the probability of a binary outcome
# here, predict risk (yes) / safe (no)
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# training
log_reg.fit(X_train_scaled, y_train)

# predictions
y_pred = log_reg.predict(X_test_scaled) # predicts class 0 (safe) or 1 (risk) for students
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1] # probability that a student belongs to class 1 - at risk


# In[31]:


print(np.mean(np.round(X_train_scaled, 2)))
print(np.std(X_train_scaled))


# In[32]:


#np.round(X_train_scaled[:5], 2)


# In[33]:


y_pred


# In[34]:


# round the probabilities
#np.set_printoptions(suppress=True)
#y_prob
# probability that a student belongs to class 1 - at risk
np.round(y_prob, 2)


# In[36]:


for prob, pred, true in zip(y_prob[:10], y_pred[:10], y_test[:10]):
    print(f"Probability risk: {prob:.2f} -> prediction: {pred} | true label: {true}")


# In[37]:


# Risk scales
for prob in y_prob[:10]:
    if prob >= 0.7:
        print(f"{prob:.2f} -> high risk")
    elif prob >= 0.3:
        print(f"{prob:.2f} -> moderate risk")
    else:
        print(f"{prob:.2f} -> low risk")


# In[38]:


# (optional) 
plt.hist(y_prob, bins=20)
plt.title("Distribution of risk probabilities")
plt.xlabel("Probability of being at risk")
plt.ylabel("Number of students")

plt.savefig("../src/distribution_risk_proba.png")
plt.show()


# In[39]:


# Display students at risk
indices = np.argsort(y_prob)[-10:]  # 10 highest probabilities

for i in indices:
    print(f"Student {i} -> probability risk: {y_prob[i]:.2f}")


# In[40]:


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# 89-90% of students are correctly classified
# but accuracy alone mat hide problems in imbalanced datasets => use other metrics


# In[41]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 51 safe students correctly predicted
# 20 risk students correctly detected

# 2 safe students wrongly flagged as risk
# 6 risk students who failed but were predicted safe


# In[42]:


# Classification report : computes Precision, Recall, F1-score, support
print(classification_report(y_test, y_pred))


# In[43]:


"""
Class 0 (safe) :
- precision : 0.89 => of all students predicted 'safe', 89% truly safe (correct 89% of the time)
- recall : 0.96 => 96% of safe students were correctly identified

Class 1 (risk) :
- precision : 0.91 => of all students predicted 'risk', 91% truly at risk
- recall : 0.77 => of all students actually at risk, model detects 77% of them (77% students at risk are well predicted)
=> problem: missing 23% of at risk students !!
"""


# In[44]:


sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("../src/confusion_matrix.png")
plt.show()


# In[45]:


### Train several different Machine Learning models: Decision Tree, Random Forest, k-NN ###
### Analyse which one works best !! ##


# In[46]:


# K-Nearest Neighbors (k-NN) model : predict student's outcome based on similar profiles
knn = KNeighborsClassifier(n_neighbors=5)

# training 
knn.fit(X_train_scaled, y_train) 

# prediction
y_pred_knn = knn.predict(X_test_scaled) 
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]


# In[47]:


#y_pred_knn


# In[48]:


y_prob_knn


# In[49]:


# Accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", accuracy_knn)


# In[50]:


# Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)


# In[51]:


# Classification report : computes Precision, Recall, F1-score, support
print(classification_report(y_test, y_pred_knn))


# In[52]:


# Decision Tree model : identify specific thresholds for intervention 
dec_tree = DecisionTreeClassifier(random_state=42)

# training
dec_tree.fit(X_train_scaled, y_train)

# prediction 
y_pred_dt = dec_tree.predict(X_test_scaled) 
y_prob_dt = dec_tree.predict_proba(X_test_scaled)[:,1]


# In[53]:


#y_pred_dt


# In[54]:


y_prob_dt 


# In[55]:


# Accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy:", accuracy_dt)


# In[56]:


# Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)


# In[57]:


# Classification report : computes Precision, Recall, F1-score, support
print(classification_report(y_test, y_pred_dt))


# In[58]:


# Random Forest model : provide the most stable and accurate prediction by consideration multiple perpectives 
ran_for = RandomForestClassifier(n_estimators=100, random_state=42)

# training 
ran_for.fit(X_train_scaled, y_train)

# prediction
y_pred_rf = ran_for.predict(X_test_scaled) 
y_prob_rf = ran_for.predict_proba(X_test_scaled)[:,1]


# In[59]:


#y_pred_rf


# In[60]:


y_prob_rf


# In[61]:


# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", accuracy_rf)


# In[62]:


# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)


# In[63]:


# Classification report : computes Precision, Recall, F1-score, support
print(classification_report(y_test, y_pred_rf))


# In[64]:


print("Model comparison :")
results = {
    "Logistic Regression": accuracy,
    "Decision Tree": accuracy_dt,
    "Random Forest": accuracy_rf,
    "KNN": accuracy_knn
}

print(results)


# In[65]:


"""
Among all evaluated models, Random Forest achieved the best overall performance: 
with accuracy = 0.91, recall for risk = 0.88 and precision for risk = 0.85, 
while K-Nearest Neighbors performed the worst, particularly in detecting at-risk student, with recall for risk = 0.54.
"""


# ### Part 6: Interpretation & Feature Importance ###

# In[67]:


### 6.1 Feature importance with Random Forest ###
# Extract feature importances # 
feature_importance_rf = pd.Series(ran_for.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importance_rf.head(10) # 10 most important features


# In[68]:


# Visualisation
top10_rf = feature_importance_rf.head(10)
plt.figure(figsize=(10,6))
top10_rf.sort_values().plot(kind="barh") # horizontal bar 
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")

plt.savefig("../src/importance_score.png")
plt.show()


# In[69]:


"""
Observation: 
G2          0.30
G1          0.21
Others:     0.02 - 0.03 

=> Conclusion: The model relies heavily on academic features, and much less on all other. 
- Student's academic trajectory (1st & 2nd period grades and past failures) is the strongest indicator of final performance. 
- Student's habit and engagement (attendance - 'absences', social activity - 'goout', 'Walc', 'heath' do influence performance,
but much more weaker than academic histoty. 
- Family background, social factors: 'Fedu', 'famrel' influence moderately to student's performance, 
suggesting a supportive environment may help but it's not the main factor. 
"""


# In[70]:


### 6.2 — Logistic Regression Coefficients ###
coef_lr = pd.Series(log_reg.coef_[0], index=X_train.columns).sort_values()  
coef_lr


# In[71]:


print("Most negative coefficients (protective factors) : ")
print(coef_lr.head(10))
# higher, much safer (reduce risk)


# In[72]:


print("Most positive coefficients (risk factors) : ")
print(coef_lr.tail(10))
# higher, more risk (increase risk) 


# In[73]:


top_positive = coef_lr.tail(10)
top_negative = coef_lr.head(10)

important_lr = pd.concat([top_negative, top_positive])

plt.figure(figsize=(10,6))
important_lr.sort_values().plot(kind="barh")
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.axvline(0, color="black", linestyle="--")

plt.savefig("../src/Logistic_regression_coefs.png")
plt.show()


# In[74]:


"""
Logistic Regression's coefficients confirm that academic performance factors play an important role in predicting 
student success, showing that higher values of G2 and G1 would strongly reduce risk, 
while behaviors such as: absences, goout,... would increase the risk. 
"""


# In[ ]:





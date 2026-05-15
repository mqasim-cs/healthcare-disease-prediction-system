from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

#mat
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
#end

#------------HEART DISEASE DATA SET-------------

#FETCHING DATA SET:
heart_disease = fetch_ucirepo(id=45) 
  
#DATA:
X_heart = heart_disease.data.features 
y_heart = heart_disease.data.targets 

#REMOVING ROWS WITH MISSING DATA:
X_heart = X_heart.dropna()
y_heart = y_heart.loc[X_heart.index]

#converting to 1D
y_heart = y_heart.values.ravel()

#Features Scaling:
scaler = StandardScaler()
X_heart_scaled = scaler.fit_transform(X_heart)

#Train test split:
X_train_heart , X_test_heart , y_train_heart , y_test_heart = train_test_split(
    X_heart_scaled , y_heart , test_size=0.2 , random_state=42)

#model training:
model_heart = LogisticRegression(max_iter = 1000)
model_heart.fit(X_train_heart , y_train_heart)
y_pred = model_heart.predict(X_test_heart)

#Accuracy:
print("Accuracy:", accuracy_score(y_test_heart, y_pred) , "\n")
print(classification_report(y_test_heart, y_pred) , "\n")


#-------------BREAST CANCER DATA SET--------------
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X_cancer = breast_cancer_wisconsin_diagnostic.data.features 
y_cancer = breast_cancer_wisconsin_diagnostic.data.targets 

y_cancer =y_cancer.values.ravel()
# Encode target variable to numeric (0=Benign, 1=Malignant)  <-- FIX
le = LabelEncoder()
y_cancer = le.fit_transform(y_cancer)

#removing missing values
X_cancer =X_cancer.dropna()


#feature scaling
scaler_cancer = StandardScaler()
X_cancer_scaled = scaler_cancer.fit_transform(X_cancer)

X_train_cancer , X_test_cancer , y_train_cancer , y_test_cancer = train_test_split(X_cancer_scaled
 , y_cancer ,test_size= 0.2  , random_state= 42)
model_cancer = LogisticRegression(max_iter=1000)
model_cancer.fit(X_train_cancer , y_train_cancer)
y_pred_cancer = model_cancer.predict(X_test_cancer)
print("Accuracy : " , accuracy_score(y_test_cancer , y_pred_cancer))
print(classification_report(y_test_cancer, y_pred_cancer) , "\n")


#------------DIABETES DATA SET-------------
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X_diabetes = cdc_diabetes_health_indicators.data.features 
y_diabetes = cdc_diabetes_health_indicators.data.targets 

#removing missing values
X_diabetes =X_diabetes.dropna()
y_diabetes = y_diabetes.loc[X_diabetes.index]

#1d array
y_diabetes = y_diabetes.values.ravel()

#feature scaling
scaler_diabetes = StandardScaler()
X_diabetes_scaled = scaler_diabetes.fit_transform(X_diabetes)

X_train_diabetes , X_test_diabetes , y_train_diabetes , y_test_diabetes = train_test_split(X_diabetes_scaled
 , y_diabetes
 ,test_size= 0.2  , random_state= 42)
model_diabetes = LogisticRegression(max_iter=1000)
model_diabetes.fit(X_train_diabetes , y_train_diabetes)
y_pred_diabetes = model_diabetes.predict(X_test_diabetes)
print("Accuracy : " , accuracy_score(y_test_diabetes , y_pred_diabetes))
print(classification_report(y_test_diabetes, y_pred_diabetes) , "\n")


#*********MATPLOT LIB*********


#heart disease

y_pred_heart = model_heart.predict(X_test_heart)
cm_heart = confusion_matrix(y_test_heart, y_pred_heart)

plt.figure(figsize=(8,6))
sns.heatmap(cm_heart, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0,1,2,3,4],
            yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Heart Disease Dataset (Multi-class)")
plt.show()

#cancer

y_pred = model_cancer.predict(X_test_cancer)
cm = confusion_matrix(y_test_cancer, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Cancer Dataset")
plt.show()

#THIS GRAPH SHOWS US:
'''Actual Benign, Predicted Benign = 70

 model correctly predicted 70 Benign cases. 

Actual Benign, Predicted Malignant = 1

1 Benign case was wrongly predicted as Malignant. 

Actual Malignant, Predicted Benign = 2

2 Malignant cases were wrongly predicted as Benign. 

Actual Malignant, Predicted Malignant = 41

41 Malignant cases were correctly predicted. 
'''
#diabetes
y_pred = model_diabetes.predict(X_test_diabetes)
cm_diabetes = confusion_matrix(y_test_diabetes, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm_diabetes, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No Diabetes','Diabetes'],
            yticklabels=['No Diabetes','Diabetes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Diabetes Dataset")
plt.show()




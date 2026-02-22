#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#load dataset
df = pd.read_csv('d:/python_ka_chilla/AI Projects/Disease Prediction (e.g., Heart Disease) using ML algorithms/heart_Disease.csv')

# display first few rows of the dataset
print("Heart Disease DataSet:\n", df.head())

#basic data preprocessing
df = df.dropna()  # remove missing values if any

#feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))

X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
y = df['target']  # target variable

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
#make predictions on the test set
y_pred_log = log_model.predict(X_test)
#evaluate the logistic regression model
print("Logistic Regression Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

#train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#make predictions on the test set
y_pred = model.predict(X_test)
#evaluate the model
print("Random Forest Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# evalute the best model
if accuracy_score(y_test, y_pred) > accuracy_score(y_test, y_pred_log):
    best_model = model
    best_y_pred = y_pred
    model_name = "Random Forest"
else:
    best_model = log_model
    best_y_pred = y_pred_log
    model_name = "Logistic Regression"
print(f"The best model is: {model_name}")

#Confusion matrix visualization for the best model
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, best_y_pred), annot=True, fmt='d', cmap='Greens', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix - {model_name}')
plt.show()

#make predictions on a new sample data
new_data = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])  # Example feature values

#scale the new data
new_data_scaled = scaler.transform(new_data)
prediction = best_model.predict(new_data_scaled)
print(f"Prediction for the new data sample: {'Disease' if prediction[0]==1 else 'No Disease'}")




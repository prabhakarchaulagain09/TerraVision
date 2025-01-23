import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

data = pd.read_csv('Crop_recommendation.csv')

# Exploratory Data Analysis (EDA)
print("Dataset Overview:")
print(data.head())
print("\nDataset Information:")
data.info()

# Visualize correlations
plt.figure(figsize=(10, 6))

# Select only numeric features for correlation calculation
numeric_data = data.select_dtypes(include=np.number)
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')

plt.title('Feature Correlation')
plt.show()

# Splitting features and target
X = data.drop('label', axis=1) # drops label
y = data['label']
print(data)
print(X)
print(y)

# Encoding the target variable
y_encoded = pd.factorize(y)[0]  # Encode crop names as integers
print(y_encoded)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) #model training

# Model Evaluation
y_pred = rf_model.predict(X_test)
#print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

mydata = [90,  42,  43,    20.879744,  82.002744,  6.502985,  202.935536]
#pd.DataFrame
mydata = np.array(mydata).reshape(1,-1)

result = rf_model.predict(mydata)
print(result)

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the Model
joblib.dump(rf_model, 'crop_recommendation_model.pkl')
print("Model saved as 'crop_recommendation_model.pkl'")



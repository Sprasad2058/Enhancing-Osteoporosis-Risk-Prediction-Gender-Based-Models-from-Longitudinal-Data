import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your dataset
data = pd.read_excel(r'C:\Users\shyam\Downloads\new delta\new_remained1_merged_data_with_all_features_and_deltas.xlsx')  # Replace with your file path
X = data.drop(columns=['Combined_SOSTEO'])  
y = data['Combined_SOSTEO']                 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the function to optimize for SVM
def svm_cv(C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
    
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set bounds for hyperparameters
params = {
    'C': (0.1, 1),       
    'gamma': (0.001, 1),  
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=svm_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=30)

# Extract the best parameters
best_params = optimizer.max['params']
model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', random_state=42)

# Train the model with the best parameters on the training set
model.fit(X_train, y_train)

# Calculate and display training accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"Training Accuracy: {train_accuracy:.6f}")

# Calculate cross-validation accuracy on the training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
cv_accuracy = cv_scores.mean()
print(f"Cross-Validation Accuracy (Training set): {cv_accuracy:.6f}")

# Calculate and display testing accuracy
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Testing Accuracy: {test_accuracy:.6f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report 
report = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:")
print(report)


# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Using Random Forest for Feature Importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Extract feature importance
feature_importances = rf_model.feature_importances_

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.decision_function(X_test))
roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

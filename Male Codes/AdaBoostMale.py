import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, precision_score, recall_score, mean_absolute_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

# Load your dataset
data = pd.read_excel(r'C:\Users\shyam\Downloads\new delta\new_remained1_merged_data_with_all_features_and_deltas.xlsx')
X = data.drop(columns=['Combined_SOSTEO'])  # Features
y = data['Combined_SOSTEO']                 # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom estimator class to integrate with scikit-learn
class CustomAdaBoost(BaseEstimator):
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.model = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Delegate to the underlying AdaBoost model
        return self.model.predict_proba(X)

# Define the function to optimize
def adaboost_cv(n_estimators, learning_rate):
    model = CustomAdaBoost(n_estimators=n_estimators, learning_rate=learning_rate)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set bounds for the hyperparameters
params = {
    'n_estimators': (50, 500),  # Number of boosting rounds (learners)
    'learning_rate': (0.01, 1.0)  # Learning rate
}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(f=adaboost_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=30)

# Extract the best parameters
best_params = optimizer.max['params']

# Train the final model with the best parameters
final_model = CustomAdaBoost(n_estimators=int(best_params['n_estimators']), learning_rate=best_params['learning_rate'])
final_model.fit(X_train, y_train)

# Calculate and display training accuracy
train_pred = final_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_accuracy:.6f}")

# Calculate cross-validation accuracy on the training set
cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
cv_accuracy = cv_scores.mean()
print(f"Cross-Validation Accuracy (Training set): {cv_accuracy:.6f}")

# Calculate and display testing accuracy
test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Testing Accuracy: {test_accuracy:.6f}")

# Predict probabilities for the test set
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred_proba.astype(int))

# Visualize confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.title("Confusion Matrix for AdaBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Calculate MAE using probabilities
mae_proba = mean_absolute_error(y_test, y_pred_proba)
print(f"Mean Absolute Error (MAE using probabilities): {mae_proba:.6f}")

# Calculate MAE using class predictions
mae_class = mean_absolute_error(y_test, (y_pred_proba > 0.5).astype(int))
print(f"Mean Absolute Error (MAE using class predictions): {mae_class:.6f}")

# Generate the classification report
report = classification_report(y_test, (y_pred_proba > 0.5).astype(int), target_names=["Class 0", "Class 1"], digits=6)
print("Classification Report:")
print(report)

# Compute ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for AdaBoost')
plt.legend(loc="lower right")
plt.show()

# Print AUC value
print(f"ROC AUC: {roc_auc:.6f}")


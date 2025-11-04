import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_excel('V1V2TrainingFiltered.xlsx')

# Separate features and target variable
X = data.drop(columns=['OR_Result'])  
y = data['OR_Result']                 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Function for cross-validation to optimize
def rf_cv(n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_features=max_features,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42,
        n_jobs=-1
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set bounds for hyperparameters
params = {
    'n_estimators': (10, 20),           
    'max_depth': (5, 10),                
    'max_features': (0.1, 0.6),         
    'min_samples_split': (2, 20),        
    'min_samples_leaf': (1, 10)          
}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(f=rf_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=6, n_iter=30)

# Extract the best parameters
best_params = optimizer.max['params']

# Train the final model with the best parameters
final_model = RandomForestClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    max_features=best_params['max_features'],
    min_samples_split=int(best_params['min_samples_split']),
    min_samples_leaf=int(best_params['min_samples_leaf']),
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train, y_train)

# Evaluate model performance

train_pred = final_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_accuracy:.6f}")

# Cross-validation accuracy
cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
cv_accuracy = cv_scores.mean()
print(f"Cross-Validation Accuracy (Training set): {cv_accuracy:.6f}")

# Testing accuracy
test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Testing Accuracy: {test_accuracy:.6f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, test_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=6))


# Plot feature importance (Top 50)
plt.figure(figsize=(10, 12)) 
feature_importances = final_model.feature_importances_
sorted_idx = feature_importances.argsort()[::-1]  

# Select the top 50 features
top_20_idx = sorted_idx[:20]
top_20_features = X_train.columns[top_20_idx]
top_20_importances = feature_importances[top_20_idx]

plt.barh(top_20_features[::-1], top_20_importances[::-1], color='skyblue')  
plt.title('Top 20 Features for Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


# Print the top 10 most important features
print("Top 10 Features:")
for i in range(10):
    print(f"{i+1}. {top_20_features[i]}: {top_20_importances[i]:.6f}")

# Predict probabilities for the test set
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

mae_proba = mean_absolute_error(y_test, y_pred_proba)
print(f"Mean Absolute Error (MAE using probabilities): {mae_proba:.6f}")

mae_class = mean_absolute_error(y_test, (y_pred_proba > 0.5).astype(int))
print(f"Mean Absolute Error (MAE using class predictions): {mae_class:.6f}")

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
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc="lower right")
plt.show()

# Print AUC value
print(f"ROC AUC: {roc_auc:.6f}")


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_excel('V1V2TrainingFiltered.xlsx')  

# Step 1: Remove Outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=42)  
'''
outliers = iso_forest.fit_predict(data.drop(columns=['OR_Result']))  # Exclude target variable
outliers = outliers == -1  # 1 for inlier, -1 for outlier

print(f"Number of outliers detected: {outliers.sum()}")

# Remove outliers from the dataset
data_cleaned = data[~outliers]
print("Dataset size after removing outliers:", data_cleaned.shape)
'''

# Define features and target variable
X = data.drop(columns=['OR_Result'])  
y = data['OR_Result']
'''

# Handle class imbalance with SMOTE (if necessary)
if y.value_counts(normalize=True).min() < 0.4:
    print("Class imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
'''

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the function to optimize for XGBoost
def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, gamma):
    model = XGBClassifier(
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        min_child_weight=min_child_weight,
        gamma=gamma,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    # Return mean cross-validation score as a metric for optimization
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set bounds for hyperparameters
params = {
    'max_depth': (3, 8),
    'learning_rate': (0, 0.02),
    'n_estimators': (100, 300),
    'min_child_weight': (7, 10),
    'gamma': (0, 5),
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=xgb_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=30)

# Extract the best parameters
best_params = optimizer.max['params']
model = XGBClassifier(
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    n_estimators=int(best_params['n_estimators']),
    min_child_weight=best_params['min_child_weight'],
    gamma=best_params['gamma'],
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

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

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], digits=6))

# Predict probabilities for the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Print AUC
print(f"ROC AUC: {roc_auc:.6f}")

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Calculate and display MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.6f}")

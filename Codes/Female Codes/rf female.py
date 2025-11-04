import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_excel(r'C:\Users\shyam\Downloads\new delta\new_remained1_merged_data_with_all_features_and_deltas.xlsx')  # Replace with your file path

# Remove Outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=42) 
outliers = iso_forest.fit_predict(data.drop(columns=['Combined_SOSTEO']))  
outliers = outliers == -1  # 1 for inlier, -1 for outlier

print(f"Number of outliers detected: {outliers.sum()}")

# Remove outliers from the dataset
data_cleaned = data[~outliers]
print("Dataset size after removing outliers:", data_cleaned.shape)

# Define features and target variable
X = data_cleaned.drop(columns=['Combined_SOSTEO']) 
y = data_cleaned['Combined_SOSTEO']

# Handle class imbalance with SMOTE 
if y.value_counts(normalize=True).min() < 0.4:
    print("Class imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the function to optimize for Random Forest
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42,
        n_jobs=-1
    )
    # Return mean cross-validation score as a metric for optimization
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set bounds for hyperparameters
params = {
    'n_estimators': (100, 300),
    'max_depth': (3, 8),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5),
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=rf_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=30)

# Extract the best parameters
best_params = optimizer.max['params']
model = RandomForestClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    min_samples_split=int(best_params['min_samples_split']),
    min_samples_leaf=int(best_params['min_samples_leaf']),
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

# Round test accuracy to two decimal places
test_accuracy_rounded = round(test_accuracy * 100, 0)

print(f"Testing Accuracy: {test_accuracy_rounded:.4f}")

# Calculate and display additional metrics
f1 = f1_score(y_test, y_pred, average='weighted')
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
mae = mean_absolute_error(y_test, y_pred)

print(f"F1 Score: {f1:.6f}")
print(f"AUC-ROC: {auc_roc:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Plot feature importance (Top 50)
plt.figure(figsize=(10, 12))  
feature_importances = model.feature_importances_
sorted_idx = feature_importances.argsort()[::-1] 

# Select the top 50 features
top_50_idx = sorted_idx[:50]
top_50_features = X_train.columns[top_50_idx]
top_50_importances = feature_importances[top_50_idx]

plt.barh(top_50_features[::-1], top_50_importances[::-1], color='skyblue') 
plt.title('Top 50 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('RandomForest_feature_importance_plot.jpeg', format='jpeg')
plt.show()

# Print the top 10 most important features
print("Top 10 Features:")
for i in range(10):
    print(f"{i+1}. {top_50_features[i]}: {top_50_importances[i]:.6f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('RandomForest_Confusion_matrics.jpeg', format='jpeg')
plt.show()

# Calculate ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('RandomForest_roc_curve.jpeg', format='jpeg')
plt.show()

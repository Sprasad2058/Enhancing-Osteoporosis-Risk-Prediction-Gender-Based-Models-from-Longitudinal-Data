import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, f1_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load dataset
data = pd.read_excel(r'C:\Users\shyam\Downloads\new delta\new_remained1_merged_data_with_all_features_and_deltas.xlsx')  # Replace with your file path

# Remove Outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=42)  
outliers = iso_forest.fit_predict(data.drop(columns=['Combined_SOSTEO']))  
outliers = outliers == -1 

print(f"Number of outliers detected: {outliers.sum()}")
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

# Feature importance
feature_importances = model.feature_importances_
sorted_idx = feature_importances.argsort()[::-1] 

# Mapping of feature names to human-readable names
feature_name_mapping = {
    'v5_HHR1': 'v5_Do you have difficulty doing heavy housework?',
    'v2_HHR1': 'v2_Do you have difficulty doing heavy housework?',
    'v5_HH1': 'v5_Can you do heavy housework?',
    'v5_WLKR1': 'v5_Difficultly In walking',
    'v5_CLBR1': 'v5_Difficultly In Climbing Step',
    'v2_HH1': 'v2_Can you do heavy housework',
    'v5_OSTFX': 'v5_vertebral fracture',
    'v2_CLBR1': 'v2_Difficult In Climbing Step',
    'delta_SHIP': 'delta_Hip Pain',
    'v2_HIPI': 'v2_Hip fracture post current visit',
    'v5_ANYTOV': 'v5_any fracture since age 50',
    'v2_WLKR1': 'v2_Difficult In walking',
    'v5_HIPI': 'v5_Hip fracture post current visit',
    'v2_ITC': 'v2_Intertrochanteric BMC',
    'v2_FALL': 'v2_Fallen or hit an object',
    'v2_HIPX': 'v2_Hip fracture post current visit',
    'v5_WLK1': 'v5_Can you walk 2-3 blocks outside?',
    'v2_ANYTOV2': 'v2_any fracture since age 50',
    'v2_BIRTH': 'v2_Have you ever given birth?',
    'v5_CLB1': 'v2_Can you climb 10 steps without stopping?',
    
}

# Replace feature names with combined labels for the plot
top_20_idx = sorted_idx[:20] 
top_20_features = X_train.columns[top_20_idx]
top_20_importances = feature_importances[top_20_idx]

# Combine original names and human-readable names
top_20_combined_labels = [
    f"{feat} ({feature_name_mapping.get(feat, 'No Mapping')})"
    for feat in top_20_features
]

# Plot the top 10 features with combined labels
plt.figure(figsize=(10, 8))  
plt.barh(top_20_combined_labels[::-1], top_20_importances[::-1], color='skyblue') 
plt.title('Top 20 Feature Importances ')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('Xgboost_top20_feature_importance_with_original_names.jpeg', format='jpeg')
plt.show()

# Print the top 10 most important features with combined labels
print("Top 10 Features:")
for i in range(10):
    readable_name = feature_name_mapping.get(top_20_features[i], 'No Mapping')
    print(f"{i+1}. {top_20_features[i]} ({readable_name}): {top_20_importances[i]:.6f}")
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


# Calculate probabilities for the positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate AUC-ROC score (explicitly)
auc_roc = auc(fpr, tpr)
print(f"AUC-ROC Score: {auc_roc:.6f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)

# Axis labels and title with increased font size
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)

# Legend with larger font
plt.legend(loc='lower right', fontsize=13)

# Optional: increase tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('Xgboost_roc_curve.jpeg', format='jpeg')
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('Xgboost_Confusion_matrics.jpeg', format='jpeg')
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_excel(r'C:\Users\shyam\Downloads\new delta\new_remained1_merged_data_with_all_features_and_deltas.xlsx')  # Replace with your file path

# Remove Outliers using Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.2, random_state=42)
outliers = iso_forest.fit_predict(data.drop(columns=['Combined_SOSTEO']))
outliers = outliers == -1  

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

# Scale features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the function to optimize AdaBoost
def ada_boost_cv(learning_rate, n_estimators, max_depth):
   
    base_estimator = DecisionTreeClassifier(max_depth=int(max_depth), random_state=42)
    model = AdaBoostClassifier(base_estimator=base_estimator, 
                               learning_rate=learning_rate, 
                               n_estimators=int(n_estimators),
                               random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Set bounds for hyperparameters
params = {
    'learning_rate': (0.01, 1),
    'n_estimators': (50, 500),
    'max_depth': (3, 20),
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=ada_boost_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=30)

# Extract the best parameters
best_params = optimizer.max['params']
print(f"Best Parameters: {best_params}")

# Train the final AdaBoost model with optimized parameters
base_estimator = DecisionTreeClassifier(max_depth=int(best_params['max_depth']), random_state=42)
model = AdaBoostClassifier(base_estimator=base_estimator, 
                           learning_rate=best_params['learning_rate'], 
                           n_estimators=int(best_params['n_estimators']),
                           random_state=42)

model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
auc_roc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
mae = mean_absolute_error(y_test, y_pred)

print(f"Testing Accuracy: {test_accuracy:.6f}")
print(f"F1 Score: {f1:.6f}")
print(f"AUC-ROC: {auc_roc:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Plot training and validation accuracy 
plt.figure(figsize=(8, 6))
plt.plot(model.estimators_, label='AdaBoost Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Estimator Importance')
plt.title('AdaBoost Estimators Importance')
plt.legend()
plt.tight_layout()
plt.savefig('AdaBoost_Estimators_Importance.jpeg', format='jpeg')
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('AdaBoost_Confusion_Matrix.jpeg', format='jpeg')
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('AdaBoost_ROC_Curve.jpeg', format='jpeg')
plt.show()

# Calculate feature importances using RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Get feature importance from Random Forest
feature_importances = rf_model.feature_importances_
features = data_cleaned.drop(columns=['Combined_SOSTEO']).columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the top 10 features
print("Top 10 Features:")
print(importance_df.head(10))

# Plot the top 50 features
top_50_features = importance_df.head(50)

plt.figure(figsize=(10, 12))
plt.barh(top_50_features['Feature'], top_50_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.title('Top 50 Features by Importance')
plt.tight_layout()
plt.savefig('Top_50_Features_Importance.jpeg', format='jpeg')
plt.show()

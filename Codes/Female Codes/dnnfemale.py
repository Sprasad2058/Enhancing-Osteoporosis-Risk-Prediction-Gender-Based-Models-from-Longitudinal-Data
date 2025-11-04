import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset----
data = pd.read_excel(r'C:\Users\shyam\Downloads\new delta\new_remained1_merged_data_with_all_features_and_deltas.xlsx')  # Replace with your file path

# Remove Outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=42)
outliers = iso_forest.fit_predict(data.drop(columns=['Combined_SOSTEO']))
outliers = outliers == -1 

print(f"Number of outliers detected: {outliers.sum()}")

# Remove outliers-------
data_cleaned = data[~outliers]
print("Dataset size after removing outliers:", data_cleaned.shape)

# Define features and target variable
X = data_cleaned.drop(columns=['Combined_SOSTEO'])  
y = data_cleaned['Combined_SOSTEO']

# Handle class imbalance with SMOTE-------- 
if y.value_counts(normalize=True).min() < 0.4:
    print("Class imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

# Split the dataset--------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function for cross-validation and training accuracy computation------
def cross_validate_dnn(model_fn, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cross_val_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        
        model = model_fn()
        history = model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        
        
        val_accuracy = np.mean(history.history['val_accuracy'])
        cross_val_scores.append(val_accuracy)
    
    return np.mean(cross_val_scores), np.std(cross_val_scores)

# Define the function to optimize DNN--------
def dnn_cv(dropout_rate, learning_rate, num_neurons):
    model = Sequential([
        Dense(int(num_neurons), activation='relu', input_dim=X_train.shape[1]),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
    val_acc = np.mean(history.history['val_accuracy'])
    return val_acc

# Set bounds for hyperparameters-------
params = {
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (0.0001, 0.01),
    'num_neurons': (32, 256),
}

# Bayesian optimization-------
optimizer = BayesianOptimization(f=dnn_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=30)

# Extract the best parameters-------
best_params = optimizer.max['params']

# Final model creation with optimized parameters-------
model = Sequential([
    Dense(int(best_params['num_neurons']), activation='relu', input_dim=X_train.shape[1]),
    Dropout(best_params['dropout_rate']),
    Dense(1, activation='sigmoid')
])

# Compile the model-----
model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model------
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Perform cross-validation and get accuracy scores-------
mean_cv_accuracy, std_cv_accuracy = cross_validate_dnn(lambda: model, X_train, y_train, n_splits=5)

# Training Accuracy
training_accuracy = history.history['accuracy'][-1]
print(f"Cross-Validation Accuracy: {mean_cv_accuracy:.6f} ± {std_cv_accuracy:.6f}")
print(f"Training Accuracy: {training_accuracy:.6f}")

# Evaluate on the test set------------
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

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

# Plot training and validation accuracy-----------
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('DNN Training and Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('DNN_training_validation_accuracy.jpeg', format='jpeg')
plt.show()

# Plot confusion matrix----------------
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('DNN_Confusion_Matrix.jpeg', format='jpeg')
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
plt.savefig('DNN_ROC_Curve.jpeg', format='jpeg')
plt.show()

# Train a Random Forest as a surrogate model to calculate feature importance
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
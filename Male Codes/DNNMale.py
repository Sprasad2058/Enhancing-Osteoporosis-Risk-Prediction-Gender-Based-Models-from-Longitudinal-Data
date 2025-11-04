import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_excel('V1V2TrainingFiltered.xlsx')  
X = data.drop(columns=['OR_Result'])  
y = data['OR_Result']                 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define the Deep Neural Network model
def create_dnn(layers, units, dropout_rate):
    model = Sequential()
    model.add(Dense(int(units), activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(dropout_rate))

    # Add additional layers as per the specified layers parameter
    for _ in range(int(layers)):
        model.add(Dense(int(units), activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))  
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define a custom estimator class to integrate with scikit-learn
class KerasDNN(BaseEstimator):
    def __init__(self, layers=1, units=64, dropout_rate=0.2, epochs=10, batch_size=32):
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        self.model = create_dnn(self.layers, self.units, self.dropout_rate)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

# Define the function to optimize
def dnn_cv(layers, units, dropout_rate, epochs, batch_size):
    model = KerasDNN(layers=int(layers), units=units, dropout_rate=dropout_rate, epochs=int(epochs), batch_size=int(batch_size))
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set bounds for the hyperparameters
params = {
    'layers': (1, 5),            
    'units': (32, 256),          
    'dropout_rate': (0.0, 0.5),  
    'epochs': (10, 100),         
    'batch_size': (16, 128)      
}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(f=dnn_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=10)

# Extract the best parameters
best_params = optimizer.max['params']

# Train the final model with the best parameters
model = KerasDNN(
    layers=int(best_params['layers']),
    units=int(best_params['units']),
    dropout_rate=best_params['dropout_rate'],
    epochs=int(best_params['epochs']),
    batch_size=int(best_params['batch_size'])
)
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

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix for DNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Calculate probabilities for the positive class
y_pred_proba = model.predict(X_test)  
y_pred_proba = y_pred_proba.flatten()  

# Calculate MAE using probabilities
mae_proba = mean_absolute_error(y_test, y_pred_proba)
print(f"Mean Absolute Error (MAE using probabilities): {mae_proba:.6f}")

# Calculate MAE using class predictions
mae_class = mean_absolute_error(y_test, (y_pred_proba > 0.5).astype(int))
print(f"Mean Absolute Error (MAE using class predictions): {mae_class:.6f}")

# Calculate ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate AUC-ROC score
auc_roc = auc(fpr, tpr)
print(f"AUC-ROC Score: {auc_roc:.6f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for DNN')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
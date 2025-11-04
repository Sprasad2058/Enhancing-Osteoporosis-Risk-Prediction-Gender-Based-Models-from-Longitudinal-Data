import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_excel('V1V2TrainingFiltered.xlsx')  
X = data.drop(columns=['OR_Result'])  
y = data['OR_Result']        
         
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the function to optimize for SVM
def svm_cv(C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
   
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set bounds for hyperparameters
params = {
    'C': (0.01, 1),       
    'gamma': (0.01, 1), 
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=svm_cv, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=30)

# Extract the best parameters
best_params = optimizer.max['params']
# Train the model with the best parameters on the training set, enabling probabilities
model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', random_state=42, probability=True)
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

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=6))
print("MAE: ", mean_absolute_error(y_test, y_pred))

# Predict probabilities for the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Print AUC value
print(f"ROC AUC: {roc_auc:.6f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

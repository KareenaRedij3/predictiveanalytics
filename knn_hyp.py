import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = '/Users/kareenaredij/Downloads/Breast_Cancer_dataset.csv'
data = pd.read_csv(file_path)

# Data preprocessing: Assuming the last column is the target
# Convert categorical data to numeric (if any) and handle NaN values
data = data.apply(pd.to_numeric, errors='ignore')
numeric_data = data.select_dtypes(include=['number'])
categorical_data = pd.get_dummies(data.select_dtypes(exclude=['number']), drop_first=True)
processed_data = pd.concat([numeric_data, categorical_data], axis=1).dropna()

# Define features and target
X = processed_data.iloc[:, :-1].values
y = processed_data.iloc[:, -1].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],         # Number of neighbors
    'metric': ['euclidean', 'manhattan']     # Distance metric
}

# Initialize KNN model
knn = KNeighborsClassifier()

# Perform Grid Search with cross-validation
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_knn.fit(X_train, y_train)

# Get the best parameters and best score from the grid search
best_knn_params = grid_search_knn.best_params_
best_knn_score = grid_search_knn.best_score_

print("Best Parameters for KNN:", best_knn_params)
print("Best Cross-Validation Accuracy for KNN:", best_knn_score)

# Evaluate on the test set using the best model from grid search
best_knn_model = grid_search_knn.best_estimator_
y_pred = best_knn_model.predict(X_test)

# Print final test accuracy and classification report
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy with Best Parameters:", test_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
file_path = '/Users/kareenaredij/Downloads/Breast_Cancer_dataset.csv'
data = pd.read_csv(file_path)

# Convert columns to numeric where possible
data = data.apply(pd.to_numeric, errors='ignore')

# Separate numeric and non-numeric columns, then one-hot encode non-numeric data
num_data = data.select_dtypes(include=['number'])
cat_data = pd.get_dummies(data.select_dtypes(exclude=['number']), drop_first=True)

# Combine numeric and encoded categorical data, dropping any remaining NaN values
processed_data = pd.concat([num_data, cat_data], axis=1).dropna()

# Define features and target (assuming target is the last column)
X = processed_data.iloc[:, :-1].values
y = processed_data.iloc[:, -1].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train KNN model
k = 5
metric = 'euclidean'
knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
knn_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

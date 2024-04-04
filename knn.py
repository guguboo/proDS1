import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data from the Excel file
data = pd.read_excel('C://mak/PRODS/dataset_satelit_latihan_1.xlsx')

# Split the data into features (spectral bands) and target label
X = data[['B2', 'B3', 'B4']]  # Features: Spectral bands B2, B4
y = data['jenis_lahan']  # Target label: Land cover types

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Predict land cover types for the test data
y_pred = knn_classifier.predict(X_test)

# Create a DataFrame to store predicted and actual labels
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the DataFrame with predicted and actual labels
print("\nPredicted and Actual Labels:")
print(results_df)

# Export the results DataFrame to an Excel file
#results_df.to_excel('prediction_results.xlsx', index=False)

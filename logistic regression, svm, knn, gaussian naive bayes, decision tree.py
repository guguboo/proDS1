import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data from the Excel file
data = pd.read_excel('C://mak/PRODS/dataset_satelit_latihan_1.xlsx')

# Split the data into features (spectral bands) and target label
X = data[['B2', 'B3', 'B4']]  # Features: Spectral bands B2, B4
y = data['jenis_lahan']  # Target label: Land cover types

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    print(f"\nTraining and evaluating {clf_name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier

# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier()

# Train and evaluate Decision Tree Classifier
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred))


from sklearn.naive_bayes import GaussianNB

# Initialize Gaussian Naive Bayes Classifier
gaussian_nb = GaussianNB()

# Train and evaluate Gaussian Naive Bayes Classifier
gaussian_nb.fit(X_train, y_train)
y_pred = gaussian_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Gaussian Naive Bayes Accuracy:", accuracy)
print("\nGaussian Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred))

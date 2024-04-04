# libs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# dataset
file_path = r"C:\Users\irani\Downloads\ML Dr. Lars\winequality-red.csv"
df = pd.read_csv(file_path, sep=';')
X = df.drop('quality', axis=1)
y = df['quality']
def log_transform(X):
    return np.log1p(X)

features_to_transform = X.columns
preprocessor = ColumnTransformer(
    transformers=[
        ('log_transform', FunctionTransformer(log_transform), features_to_transform),
        ('std_scaler', StandardScaler(), features_to_transform)
    ]
)
# classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}
# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# eval and train clsifr
results = {}
for name, clf in classifiers.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    results[name] = (mean_accuracy, accuracy_test)
    print(f"{name}: Mean CV Accuracy: {mean_accuracy:.4f}, Test Accuracy: {accuracy_test:.4f}")

# plot
plt.figure(figsize=(12, 6))
classifier_names = list(results.keys())
mean_cv_accuracies = [x[0] for x in results.values()]
test_accuracies = [x[1] for x in results.values()]
index = np.arange(len(classifier_names))
bar_width = 0.3
bars = plt.bar(index, mean_cv_accuracies, bar_width, label='Mean CV Accuracy', color="teal")
barss = plt.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy', color="lightgray")
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Classifier Evaluation')
plt.xticks(index + bar_width / 2, classifier_names)
plt.legend()
plt.tight_layout()
plt.show()



'''''
results
Random Forest: Mean CV Accuracy: 0.6779, Test Accuracy: 0.6469
Support Vector Machine: Mean CV Accuracy: 0.5943, Test Accuracy: 0.5750
Logistic Regression: Mean CV Accuracy: 0.6044, Test Accuracy: 0.5719

'''
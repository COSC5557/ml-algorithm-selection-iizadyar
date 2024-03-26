import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

# dataset
file_path = r"C:\Users\irani\Downloads\ML Dr. Lars\winequality-red.csv"
df_red = pd.read_csv(file_path, sep=';')
X = df_red.drop('quality', axis=1)
y = df_red['quality']

# preprocessing
def log_transform(X):
    return np.log1p(X)

features_to_transform = X.columns
preprocessor = ColumnTransformer(
    transformers=[
        ('log_transform', FunctionTransformer(log_transform), features_to_transform),
        ('std_scaler', StandardScaler(), features_to_transform)
    ]
)
X_preprocessed = preprocessor.fit_transform(X)

# classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))  # Increased max_iter
]

# evaluate
results = {}
for name, clf in classifiers:
    scores = cross_val_score(clf, X_preprocessed, y, cv=10)
    results[name] = scores
    print(f"{name} Mean Accuracy: {np.mean(scores):.4f}")

plt.figure(figsize=(10, 6))
plt.violinplot(results.values(), showmeans=True)
plt.title('Comparison of Classifier Performance')
plt.ylabel('Accuracy')
plt.xlabel('Classifier')
plt.xticks(range(1, len(results) + 1), results.keys() )
plt.grid(axis='y')
plt.show()

# best classifier
best_mean_score = 0
best_classifiers = []
for name, scores in results.items():
    mean_score = np.mean(scores)
    if mean_score > best_mean_score:
        best_mean_score = mean_score
        best_classifiers = [name]
    elif mean_score == best_mean_score:
        best_classifiers.append(name)
print("Best classifier(s):", best_classifiers)
print("Mean accuracy of the best classifier(s):", best_mean_score)

'''
results:Random Forest Mean Accuracy: 0.5773
Support Vector Machine Mean Accuracy: 0.5929
Logistic Regression Mean Accuracy: 0.5891
Best classifier(s): ['Support Vector Machine']
Mean accuracy of the best classifier(s): 0.592869496855346
'''
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)

df = pd.DataFrame({
    'Age': np.random.randint(18, 70, 50000),
    'Income': np.random.normal(50000, 15000, 50000).astype(int),
    'Gender': np.random.choice([0, 1], 50000),
    'Emergency_Admissions': np.random.randint(0, 50, 50000),
    'Weekend': np.random.choice([0, 1], 50000),
})
df['Purchased'] = ((df['Gender'] == 0) & (df['Emergency_Admissions'] > 25)).astype(int)
print(df.head(5))


X = df.drop(columns=['Purchased'])
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train)

##Train random forest
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

print(clf.fit(X_train, y_train))

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("y_pred:",y_pred)
print("y_proba:",y_proba)


##Confusion matrix -
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)

print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
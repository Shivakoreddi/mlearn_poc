import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


##Load data

data = pd.read_csv('synthetic_email_spam.csv')
X = data.drop(columns=["spam"])
y = data["spam"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=65)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))



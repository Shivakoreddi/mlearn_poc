from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=400, noise=0.25, random_state=42)

model = make_pipeline(
    StandardScaler(),      # SVM hates unscaled data
    SVC(kernel='rbf', C=10, gamma=0.7)
)
model.fit(X, y)
print("Train accuracy:", accuracy_score(y, model.predict(X)))

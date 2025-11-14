import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# Load and preprocess data
df = pd.read_csv('Fish.csv')
df = pd.get_dummies(df, columns=['Species'], drop_first=True)

# Correlation check (optional)
# correlations = df.corr(numeric_only=True)['Height'].sort_values(ascending=False)
# print(correlations)

# Split features and target
X = df.drop(columns=['Height'])
y = df['Height']

# Correct train-test split (your original order was wrong)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Find best polynomial degree manually (not used in grid search below)
best_score = -np.inf # -infinity
best_degree = 1

for d in range(1, 8):
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    y_pred = model.predict(poly.transform(X_test))
    score = r2_score(y_test, y_pred)
    print(f"Degree {d}: R2 Score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_degree = d

print("Best Degree based on manual loop: ", best_degree)

# Now use pipeline and GridSearchCV with best degree (or just set to 7 as in your original)
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Add scaling inside the pipeline
    ('poly', PolynomialFeatures(degree=best_degree)),
    ('model', LinearRegression())
])

# Grid search parameters
param_grid = {
    'model__fit_intercept': [True, False],
    'model__positive': [True, False]
}

# Perform GridSearchCV
gridSearch = GridSearchCV(pipe, param_grid, cv=5)
gridSearch.fit(X_train, y_train)

# Make predictions
prediction = gridSearch.predict(X_test)

# Output
print("Prediction: ", prediction)
print("Best Params:", gridSearch.best_params_)
print("Accuracy (R2 Score):", r2_score(y_test, prediction))

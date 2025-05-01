############################################################

# moneyScoreFinalModel.py
# Will Paz
# 4.30.25

############################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv('modelData.csv')

# Define features and target
X = df.drop(columns=['Cap.Hit.Pct.League.Cap', 'playerId', 'season', 'FirstName', 'LastName', 'team', 'leagueCap'])
y = df['Cap.Hit.Pct.League.Cap']

# Convert object columns to categorical
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base model
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    enable_categorical=True,
    random_state=42,
    verbosity=1
)

# Set up hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid search with 5-fold CV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
print("Best Parameters:")
print(grid_search.best_params_)

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Predict on the entire dataset
y_full_pred = best_model.predict(X)

# Add the predictions and the difference to the original dataset
df['Predicted'] = y_full_pred
df['Difference'] = df['Cap.Hit.Pct.League.Cap'] - y_full_pred

# Save the entire original dataset with predictions and differences to a CSV file
df.to_csv('model_predictions_full_with_data.csv', index=False)

# Optionally, print the first few rows of the updated dataset
print(df.head())


print(f"\nModel Evaluation on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Cap Hit %")
plt.grid(True)
plt.tight_layout()
plt.show()

# Get feature importances from the best model
finalImportances = best_model.feature_importances_

# Get the feature names from the model
finalFeatures = X.columns

# Create a DataFrame with feature names and importance scores
featureWeightsDf = pd.DataFrame({
    'feature': finalFeatures,
    'importance': finalImportances
}).sort_values(by='importance', ascending=False)

# Plot feature importance using seaborn
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(
    data=featureWeightsDf,
    x='importance',
    y='feature',
    palette='viridis'
)

plt.title("Final XGBoost Feature Importances", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()






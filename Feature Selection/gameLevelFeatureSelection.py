import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("predictionGameData2425.csv")
print(df.columns)

featureCols = [
    'goals', 'assists', 'points', 'pim',
    'hits', 'sog', 'faceOffWinningPctg', 'blockedShots', 'giveaways',
    'takeaways', 'toiInMinutes', 'shootingPerc', 'pointsPer60',
    'goalsPer60', 'assistsPer60', 'pimPer60', 'hitsPer60', 'sogPer60',
    'blockedShotsPer60', 'giveawaysPer60', 'takeawaysPer60'
]

# featureCols = [
#     'goals', 'assists', 'points', 'plusMinus', 'pim', 'hits',
#     'powerPlayGoals', 'sog', 'faceoffWinningPctg', 'blockedShots', 'toiInMinutes',
#     'giveaways', 'takeaways', 'shootingPerc', 'pointsPer60', 'goalsper60',
#     'assistsper60', 'pimper60', 'hitsper60', 'sogper60', 'blockedShotsper60',
#     'giveawaysper60', 'takeawaysper60'
# ]

targetCol = 'Cap.Hit.Pct.League.Cap'

X = df[featureCols]
y = df[targetCol]

# Standardize data for Lasso and Ridge
standardScaler = StandardScaler()
xScaled = standardScaler.fit_transform(X)

# -------------------------------------
# 1. Extra Trees
extraTreesModel = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
extraTreesModel.fit(X, y)
extraTreesSelector = SelectFromModel(extraTreesModel, prefit=True)
extraTreesFeatures = set(X.columns[extraTreesSelector.get_support()])
print("Extra Tree Complete!")

# -------------------------------------
# 2. XGBoost
xgbModel = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgbModel.fit(X, y)
xgbSelector = SelectFromModel(xgbModel, prefit=True)
xgbFeatures = set(X.columns[xgbSelector.get_support()])
print("XGBoost Complete!")

# -------------------------------------
# 3. Lasso (L1)
from sklearn.impute import SimpleImputer

# Impute missing values before scaling
imputer = SimpleImputer(strategy='mean')
xImputed = imputer.fit_transform(X)

# Now scale
scaler = StandardScaler()
xScaled = scaler.fit_transform(xImputed)

lassoModel = LassoCV(cv=5, random_state=42).fit(xScaled, y)
lassoSelector = SelectFromModel(lassoModel, prefit=True)
lassoFeatures = set(X.columns[lassoSelector.get_support()])
print("Lasso Complete!")

# -------------------------------------
# 4. Recursive Feature Elimination (RFE)
rfeBaseModel = Ridge()
rfeSelector = RFE(estimator=rfeBaseModel, n_features_to_select=10)
rfeSelector.fit(xScaled, y)
rfeFeatures = set(X.columns[rfeSelector.support_])
print("RFE Complete!")

# -------------------------------------
# 5. Ridge Regression (L2)
ridgeModel = Ridge(alpha=1.0)
ridgeModel.fit(xScaled, y)
ridgeCoefficients = np.abs(ridgeModel.coef_)
ridgeFeatures = set(X.columns[np.argsort(ridgeCoefficients)[-10:]])  # top 10
print("Ridge Complete!")

# -------------------------------------
# Voting System
selectedFeatureSets = [extraTreesFeatures, xgbFeatures, lassoFeatures, rfeFeatures, ridgeFeatures]

featureVoteCounter = Counter(f for featureSet in selectedFeatureSets for f in featureSet)
votedFeatures = [f for f, voteCount in featureVoteCounter.items() if voteCount >= 3]

print("\n Features selected by voting (≥3 votes):")
print(votedFeatures)

# Optional: See votes per feature
print("\n Full feature vote counts:")
for feature, voteCount in featureVoteCounter.most_common():
    print(f"{feature}: {voteCount} votes")
    
# Save reduced df
baseCols = ['season', 'gameDate', 'playerId', 'LastName', 'FirstName', 'team', 'position', 'Cap.Hit.Pct.League.Cap']
filtereddf = df[baseCols + votedFeatures].copy()

# # Save reduced df
# baseCols = ['season', 'gameID', 'playerId', 'LastName', 'FirstName', 'team', 'pos', 'Cap.Hit.Pct.League.Cap']
# filtereddf = df[baseCols + votedFeatures].copy()

# Scale selected features to [0, 1]
minMaxScaler = MinMaxScaler()
filtereddf[votedFeatures] = minMaxScaler.fit_transform(filtereddf[votedFeatures])

# Create a weighted score using Extra Trees (on voted features only)
xgbModelReduced = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgbModelReduced.fit(filtereddf[votedFeatures], y)
featureWeights = xgbModelReduced.feature_importances_ / np.sum(xgbModelReduced.feature_importances_)

filtereddf['localMoneyScore'] = (filtereddf[votedFeatures] * featureWeights).sum(axis=1)

# Save output
filtereddf.to_csv("predLocalMoneyScore.csv", index=False)

# Get final voted features
finalFeatures = votedFeatures

# Get feature importances from final model
finalImportances = xgbModelReduced.feature_importances_
featureWeightsDf = pd.DataFrame({
    'feature': finalFeatures,
    'importance': finalImportances
}).sort_values(by='importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(
    data=featureWeightsDf,
    x='importance',
    y='feature',
    palette='viridis'
)

plt.title("Final XGBoost Feature Importances (Post-Voting)", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()


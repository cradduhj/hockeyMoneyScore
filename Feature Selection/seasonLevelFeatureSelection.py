############################################################

# seasonLevelFeatureSelection.py
# Will Paz
# 4.15.25

############################################################

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("predictionSeasonData2425.csv")
print(df.columns)

# Feature and target columns
# featureCols = [
#     'TOI', 'CF/60', 'CA/60', 'CF%', 'FF/60', 'FA/60', 'FF%',
#     'SF/60', 'SA/60', 'SF%', 'GF/60', 'GA/60', 'GF%', 'xGF/60',
#     'xGA/60', 'xGF%', 'SCF/60', 'SCA/60', 'SCF%', 'HDCF/60', 'HDCA/60',
#     'HDCF%', 'HDGF/60', 'HDGA/60', 'HDGF%', 'MDCF/60', 'MDCA/60',
#     'MDCF%', 'MDGF/60', 'MDGA/60', 'MDGF%', 'LDCF/60',
#     'LDCA/60', 'LDCF%', 'LDGF/60', 'LDGA/60', 'LDGF%', 'On-Ice SH%',
#     'On-Ice SV%', 'PDO', 'Off. Zone Starts/60', 'Neu. Zone Starts/60',
#     'Def. Zone Starts/60', 'Off. Zone Start %', 'Off. Zone Faceoffs/60',
#     'Neu. Zone Faceoffs/60', 'Def. Zone Faceoffs/60', 'Off. Zone Faceoff %'
# ]

featureCols = [
    'TOI', 'CF.60', 'CA.60', 'CF.', 'FF.60', 'FA.60', 'FF.', 'SF.60', 'SA.60', 'SF.',
    'GF.60', 'GA.60', 'GF.', 'xGF.60', 'xGA.60', 'xGF.', 'SCF.60', 'SCA.60',
    'SCF.', 'HDCF.60', 'HDCA.60', 'HDCF.', 'HDGF.60', 'HDGA.60', 'HDGF.',
    'MDCF.60', 'MDCA.60', 'MDCF.', 'MDGF.60', 'MDGA.60', 'MDGF.', 'LDCF.60',
    'LDCA.60', 'LDCF.', 'LDGF.60', 'LDGA.60', 'LDGF.', 'On.Ice.SH.',
    'On.Ice.SV.', 'PDO', 'Off..Zone.Starts.60', 'Neu..Zone.Starts.60',
    'Def..Zone.Starts.60', 'Off..Zone.Start..',
    'Off..Zone.Faceoffs.60', 'Neu..Zone.Faceoffs.60',
    'Def..Zone.Faceoffs.60', 'Off..Zone.Faceoff..'
]

targetCol = 'Cap.Hit.Pct.League.Cap'
X = df[featureCols]
y = df[targetCol]

# Preprocess for Lasso/Ridge
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
XImputed = imputer.fit_transform(X)
scaler = StandardScaler()
XScaled = scaler.fit_transform(XImputed)

# ------------------ MODEL 1: Extra Trees ------------------
extraTreesModel = ExtraTreesRegressor(n_estimators=50, random_state=42)
extraTreesModel.fit(X, y)
etSelector = SelectFromModel(extraTreesModel, prefit=True)
etFeatures = set(X.columns[etSelector.get_support()])

# ------------------ MODEL 2: XGBoost ------------------
xgbModel = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgbModel.fit(X, y)
xgbSelector = SelectFromModel(xgbModel, prefit=True)
xgbFeatures = set(X.columns[xgbSelector.get_support()])

# ------------------ MODEL 3: Lasso ------------------
lassoModel = LassoCV(cv=5, random_state=42).fit(XScaled, y)
lassoSelector = SelectFromModel(lassoModel, prefit=True)
lassoFeatures = set(X.columns[lassoSelector.get_support()])

# ------------------ MODEL 4: RFE ------------------
rfeModel = RFE(estimator=Ridge(), n_features_to_select=10)
rfeModel.fit(XScaled, y)
rfeFeatures = set(X.columns[rfeModel.support_])

# ------------------ MODEL 5: Ridge ------------------
ridgeModel = Ridge()
ridgeModel.fit(XScaled, y)
ridgeCoeffs = np.abs(ridgeModel.coef_)
ridgeFeatures = set(X.columns[np.argsort(ridgeCoeffs)[-10:]])

# ------------------ VOTING SYSTEM ------------------
selectedSets = [etFeatures, xgbFeatures, lassoFeatures, rfeFeatures, ridgeFeatures]
voteCounts = Counter(f for s in selectedSets for f in s)
votedFeatures = [f for f, v in voteCounts.items() if v >= 3]

print("\nSelected features (≥3 votes):")
print(votedFeatures)

print("\nFull vote counts:")
for feature, votes in voteCounts.most_common():
    print(f"{feature}: {votes} votes")

# ------------------ FINAL MODEL & SCORE ------------------
baseCols = ['season', 'name', 'team', 'Position', 'Cap.Hit.Pct.League.Cap']
filteredDf = df[baseCols + votedFeatures].copy()

# Normalize selected features
minMaxScaler = MinMaxScaler()
filteredDf[votedFeatures] = minMaxScaler.fit_transform(filteredDf[votedFeatures])

# Final XGBoost model on voted features
finalModel = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
finalModel.fit(filteredDf[votedFeatures], y)
finalWeights = finalModel.feature_importances_ / finalModel.feature_importances_.sum()

# Weighted score
filteredDf['globalMoneyScore'] = (filteredDf[votedFeatures] * finalWeights).sum(axis=1)

# Sort by globalMoneyScore
filteredDf = filteredDf.sort_values(by='globalMoneyScore', ascending=False)

# Save result
# filteredDf.to_csv("globalMoneyScore.csv", index=False)

filteredDf.to_csv("predGlobalMoneyScore.csv", index=False)

# ------------------ PLOTTING ------------------
featureWeightDf = pd.DataFrame({
    'feature': votedFeatures,
    'importance': finalWeights
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=featureWeightDf, x='importance', y='feature', palette='viridis')
plt.title("Final XGBoost Feature Importances (Post-Voting)", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()


import pandas as pd
import itertools
from elo import Elo  # Make sure this is your refined version with no homefield support

# Load data
df = pd.read_csv("localMoneyScore.csv")
seasons = df['season'].unique()

# Define parameter grid for tuning
kValues = [10, 20, 30]
gValues = [0.5, 1, 2]

results = []

# Grid search over all (k, g) combinations
for k, g in itertools.product(kValues, gValues):
    correct = 0
    total = 0

    for season in seasons:
        seasonData = df[df['season'] == season]
        elo = Elo(k=k, g=g)

        for gameId, gameGroup in seasonData.groupby('gameID'):
            players = gameGroup[['playerId', 'localMoneyScore']]

            for p1, p2 in itertools.combinations(players.itertuples(index=False), 2):
                id1, score1 = p1
                id2, score2 = p2

                if score1 == score2:
                    continue

                actualWinner = id1 if score1 > score2 else id2
                actualLoser = id2 if score1 > score2 else id1

                rating1 = elo.getRating(id1)
                rating2 = elo.getRating(id2)

                predictedWinner = id1 if rating1 > rating2 else id2
                if predictedWinner == actualWinner:
                    correct += 1
                total += 1

                elo.gameOver(actualWinner, actualLoser)

    accuracy = correct / total if total else 0
    results.append({'k': k, 'g': g, 'accuracy': accuracy})
    print(f"Tested k={k}, g={g} → Accuracy: {accuracy:.4f}")

# Results summary
resultsDf = pd.DataFrame(results)
top5 = resultsDf.sort_values(by='accuracy', ascending=False).head()
print("\nTop 5 parameter sets:")
print(top5)

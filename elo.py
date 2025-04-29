############################################################

# elo.py
# Will Paz
# 4.15.25

############################################################

import pandas as pd
import itertools
from collections import defaultdict

class Elo:
    def __init__(self, k=30, g=2, baseRating=1500, verbose=False):
        self.ratingDict = defaultdict(lambda: baseRating)
        self.k = k
        self.g = g
        self.verbose = verbose

    def gameOver(self, winner, loser):
        winnerRating = self.ratingDict[winner]
        loserRating = self.ratingDict[loser]

        expected = self.expectResult(winnerRating, loserRating)
        delta = (self.k * self.g) * (1 - expected)

        self.ratingDict[winner] += delta
        self.ratingDict[loser] -= delta

        if self.verbose:
            print(f"{winner} beat {loser} | Δ: {delta:.2f} | New Ratings → {winner}: {self.ratingDict[winner]:.1f}, {loser}: {self.ratingDict[loser]:.1f}")

    def expectResult(self, player1Rating, player2Rating):
        exponent = (player2Rating - player1Rating) / 400.0
        return 1 / (10.0 ** exponent + 1)

    def getRating(self, name):
        return self.ratingDict[name]


    def getAllRatings(self):
        return dict(self.ratingDict)

# # Load data
# df = pd.read_csv("localMoneyScore.csv")

# Load data
df = pd.read_csv("predLocalMoneyScore.csv")

# Process each season
for season in df['season'].unique():
    print(f"Processing season {season}...")

    seasonData = df[df['season'] == season]
    elo = Elo(k=30, g=2)

    # groupedGames = seasonData.groupby('gameID')
    
    groupedGames = seasonData.groupby('gameDate')

    for gameId, gameGroup in groupedGames:
        players = gameGroup[['playerId', 'team', 'localMoneyScore']]

        for p1, p2 in itertools.combinations(players.itertuples(index=False), 2):
            player1Id, team1, score1 = p1
            player2Id, team2, score2 = p2

            if score1 == score2:
                continue

            winner = player1Id if score1 > score2 else player2Id
            loser = player2Id if score1 > score2 else player1Id

            elo.gameOver(winner, loser)

    # Final ratings
    ratings = pd.DataFrame(list(elo.getAllRatings().items()), columns=['playerId', 'rating'])

    # playerInfo = seasonData[['season', 'playerId', 'FirstName', 'LastName', 'pos']].drop_duplicates()
    playerInfo = seasonData[['season', 'playerId', 'FirstName', 'LastName', 'position']].drop_duplicates()
    ratings = ratings.merge(playerInfo, on='playerId', how='left')
    ratings = ratings.sort_values(by='rating', ascending=False)

    # ratings.to_csv(f"elo_ratings_{season}.csv", index=False)
    # print(f"Saved elo_ratings_{season}.csv")
    
    ratings.to_csv(f"pred_elo_ratings_{season}.csv", index=False)
    print(f"Saved pred_elo_ratings_{season}.csv")

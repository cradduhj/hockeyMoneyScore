###############################################################################
# Joey Endres
# 418 NHL Class Project
# Feb 11
###############################################################################

install.packages("httr")

library(dplyr)
library(tidyr)
library(jsonlite)
library(httr)

# Define parameters
player_id <- 8478402   # Example: Nathan MacKinnon
season <- "20232024"   # Format: YYYYYYYY (e.g., "20232024" for 2023-24 season)
game_type <- 2         # 2 = Regular Season, 3 = Playoffs

# Construct the full API URL
url <- paste0(base_url, "/", player_id, "/game-log/", season, "/", game_type)

# Send GET request
response <- GET(url)

# Convert response to text
data <- content(response, as = "text", encoding = "UTF-8")

# Parse JSON
json_data <- fromJSON(data, flatten = TRUE)

# View JSON structure
str(json_data)

# Extract the game log data (assuming it's inside a "data" field)
game_log <- json_data$gameLog

# Convert to a dataframe
game_log_df <- as.data.frame(game_log)

# View the first few rows
head(game_log_df)
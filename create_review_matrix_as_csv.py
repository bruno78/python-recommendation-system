import pandas as pd
import numpy as np

# Read the dataset into a data table using Pandas
df = pd.read_csv("data/movie_ratings_data_set.csv")

# Convert the running list of user ratings into a matrix using the 'pivot table' function
ratings_df = pd.pivot_table(df, index="user_id", columns="movie_id", aggfunc=np.max)

# Create a CSV file of the data for easy viewing
ratings_df.to_csv("data/review_matrix.csv", na_rep="")

import numpy as np
import pandas as pd
import matrix_factorization_utilities as mfu

# Load user ratings
raw_dataset_df = pd.read_csv("data/movie_ratings_data_set.csv")

# Convert the running list of user ratings into a matrix using the 'pivot table' function.
# For repeated values (in this case if a user rated the movie twice), we are picking the highest value.
# If you want to use the mean, in aggfunc choose np.min.
ratings_df = pd.pivot_table(raw_dataset_df, index='user_id',
                            columns='movie_id', aggfunc=np.max)

# Apply matrix factorization to find the latent features.
# The result of the function will be a U matrix and a M matrix that has 15 attributes
# for each user and each movie respectively.
U, M = mfu.low_rank_matrix_factorization(ratings_df.as_matrix(),
                                                       num_features=15,
                                                       regularization_amount=0.1)

# Find all predicted ratings by multiplying the U by M
predicted_ratings = np.matmul(U, M)

# Save all the ratings to a CSV file
predicted_ratings_df = pd.DataFrame(index=ratings_df.index,
                                    columns=ratings_df.columns,
                                    data=predicted_ratings)

predicted_ratings_df.to_csv("data/predicted_ratings.csv")

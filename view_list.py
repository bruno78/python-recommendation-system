import pandas as pd
import numpy as np
import os
import webbrowser

# Read the dataset into a data table using Pandas
data_table = pd.read_csv('data/movies.csv', index_col="movie_id" )

# Create a web page view of the data for easy viewing
html = data_table.to_html()

# Save the html to a temporary file
with open("views/movie_list.html", "w") as f:
    f.write(html)

# Open the webpage in the webbrowser
full_filename = os.path.abspath("movie_list.html")
webbrowser.open("file://{}".format(full_filename))

'''
CRISP-ML(Q) process model describes six phases:
# - Business and Data Understanding
# - Data Preparation
# - Model Building
# - Model Evaluation and Hyperparameter Tuning
# - Model Deployment
# - Monitoring and Maintenance

# Business Problem: One of the e-commerce customer is facing this issue: 
    Number of products sold because of cross selling and up selling has reduced significantly

# Objective(s): Maximize Product Sales
# Constraint(s): Minimize the Coupon Fatigue

Success Criteria:
    a. Business: Increase the sales by 18% to 20%
    b. ML: 
    c. Economic: Additional revenue of $250K to $300K
    
    Data Collection: 
        Dimension: 12294 rows and 7 columns
        1. anime_id
        2. name
        3. genre
        4. type
        5. episodes
        6. rating
        7. members   
'''

# Importing all required libraries, modules
# Import the os module for operating system related functionality
import os

# Import pandas library for data manipulation
import pandas as pd

# Import TfidfVectorizer from scikit-learn for text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer 

# Import cosine_similarity from scikit-learn for calculating cosine similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity

# Import joblib for saving and loading models
import joblib

# Load the anime dataset from a CSV file
anime = pd.read_csv(r"anime.csv", encoding='utf8')

# Database Connection using SQLAlchemy
from sqlalchemy import create_engine
from urllib.parse import quote
# Define the connection string for the SQLAlchemy engine
# This connects to a MySQL database named 'recommenddb' with username 'user1' and password 'user1'
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root", pw=quote("Sai@123kumar"), db="association_rules"))

# Upload the 'anime' DataFrame to the 'anime' table in the database
# If the table already exists, it will be replaced
# Data is inserted in chunks of 1000 rows at a time
anime.to_sql('anime', con=engine, if_exists='replace', chunksize=1000, index=False)


# Define the SQL query to select all records from the 'anime' table
sql = 'select * from anime'

# Execute the SQL query and read the result into a pandas DataFrame
anime = pd.read_sql_query(sql, con=engine)

# Check for missing values in the 'genre' column
anime["genre"].isnull().sum()

# Impute missing values in the 'genre' column with 'General'
anime["genre"] = anime["genre"].fillna("General")

# Create a TfidfVectorizer object with English stop words
tfidf = TfidfVectorizer(stop_words="english")

# Transform the 'genre' column into a TF-IDF matrix
tfidf_matrix = tfidf.fit(anime.genre)

# Save the TF-IDF matrix using joblib
joblib.dump(tfidf_matrix, 'matrix')

# Get the current working directory
os.getcwd()

# Load the saved TF-IDF matrix using joblib
mat = joblib.load("matrix")

# Transform the 'genre' column of the anime DataFrame into a TF-IDF matrix using the loaded matrix
tfidf_matrix = mat.transform(anime.genre)

# Get the shape of the TF-IDF matrix
tfidf_matrix.shape 

# cosine(x, y)= (x.y) / (||x||.||y||)
# Computing the cosine similarity on Tfidf matrix

# Calculate the cosine similarity matrix between the TF-IDF vectors of anime genres
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a mapping of anime name to index number for easy lookup
anime_index = pd.Series(anime.index, index=anime['name']).drop_duplicates()

# Example: Get the index of the anime "Assassins (1995)" using the anime_index mapping
anime_id = anime_index["Assassins (1995)"]

# Display the index of the anime "Assassins (1995)"
anime_id

# Define the number of top recommendations to generate
topN = 5

# Define a function called get_recommendations with parameters Name and topN
def get_recommendations(Name, topN):

    # Getting the movie index using its title
    anime_id = anime_index[Name]  # Get the index of the movie named "Name" from the anime_index dictionary

    # Getting the pair wise similarity score for all the anime's with that anime
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))  # Create a list of tuples where the first element is the index and the second element is the cosine similarity score between the movie with index "anime_id" and all other movies in the cosine_sim_matrix

    # Sorting the cosine_similarity scores based on scores
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)  # Sort the cosine_scores list in descending order based on the similarity score

    # Get the scores of top N most similar movies
    cosine_scores_N = cosine_scores[0:topN + 1]  # Get the top N+1 (including the original movie) cosine scores from the sorted list

    # Getting the movie index
    anime_idx = [i[0] for i in cosine_scores_N]  # Extract the movie indexes from the top N+1 cosine scores
    anime_scores = [i[1] for i in cosine_scores_N]  # Extract the similarity scores from the top N+1 cosine scores

    # Similar movies and scores
    anime_similar_show = pd.DataFrame(columns=["name", "Score"])  # Create a DataFrame to store the recommended movies and their scores

    anime_similar_show["name"] = anime.loc[anime_idx, "name"]  # Fill the "name" column of the DataFrame with the names of the movies corresponding to the indexes in anime_idx
    anime_similar_show["Score"] = anime_scores  # Fill the "Score" column of the DataFrame with the similarity scores in anime_scores

    anime_similar_show.reset_index(inplace=True)  # Reset the index of the DataFrame

    return (anime_similar_show.iloc[1:, ])  # Return the DataFrame excluding the first row (which is the original movie)


# Call the custom function to make recommendations for "No Game No Life Movie" with top 10 recommendations
rec = get_recommendations("No Game No Life Movie", topN=10)

# Print the recommendations (DataFrame)
print(rec)

# Import necessary libraries
from flask import Flask, render_template, request  # Web framework for building the app
import pandas as pd  # Data manipulation library
import joblib  # Library for saving and loading machine learning models
from sqlalchemy import create_engine  # Library for connecting to databases
from sklearn.metrics.pairwise import cosine_similarity  # Library for calculating similarity scores
from urllib.parse import quote
# Database connection details (replace with your actual credentials)
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw=quote("Sai@123kumar"), db="association_rules"))

# SQL query to retrieve anime data from the database
sql = 'select * from anime'

# Read the anime data from the database using pandas
anime = pd.read_sql_query(sql, engine)

# Alternative: Read anime data from a CSV file (commented out)
# anime = pd.read_csv("anime.csv", encoding = 'utf8')

# Fill missing values in the "genre" column with empty strings
anime["genre"] = anime["genre"].fillna(" ")

# Save the anime names to a list for later use (e.g., dropdown in a web app)
movies_list = anime['name'].to_list()

# Load the pre-trained TF-IDF model from a serialized file
tfidf = joblib.load('matrix')

# Transform the anime genres using the loaded TF-IDF model
tfidf_matrix = tfidf.transform(anime.genre)

# Calculate cosine similarity matrix between all anime based on their TF-IDF representations
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series to map anime names to their corresponding indices in the DataFrame (ensuring unique mapping)
anime_index = pd.Series(anime.index, index=anime['name']).drop_duplicates()

def get_recommendations(Name, topN):
  """
  This function takes a movie title (Name) and the desired number of recommendations (topN) as input.
  It then retrieves the most similar movies based on cosine similarity and returns a DataFrame containing their names and similarity scores.
  """

  # Get the index of the movie named "Name" from the anime_index dictionary
  anime_id = anime_index[Name]

  # Create a list of tuples where the first element is the index and the second element is the cosine similarity score
  # between the movie with index "anime_id" and all other movies in the cosine_sim_matrix
  cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))

  # Sort the cosine_scores list in descending order based on the similarity score (highest similarity first)
  cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)

  # Get the top N+1 (including the original movie) cosine scores from the sorted list
  cosine_scores_N = cosine_scores[0:topN + 1]

  # Extract the movie indexes from the top N+1 cosine scores
  anime_idx = [i[0] for i in cosine_scores_N]

  # Extract the similarity scores from the top N+1 cosine scores
  anime_scores = [i[1] for i in cosine_scores_N]

  # Create a DataFrame to store the recommended movies and their scores
  anime_similar_show = pd.DataFrame(columns=["name", "Score"])

  # Fill the "name" column of the DataFrame with the names of the movies corresponding to the indexes in anime_idx
  anime_similar_show["name"] = anime.loc[anime_idx, "name"]

  # Fill the "Score" column of the DataFrame with the similarity scores in anime_scores
  anime_similar_show["Score"] = anime_scores

  # Reset the index of the DataFrame (optional, can be useful for further manipulation)
  anime_similar_show.reset_index(inplace=True)

  # Return the DataFrame excluding the first row (which is the original movie)
  return anime_similar_show.iloc[1:, ]

# Initialize a Flask application instance
app = Flask(__name__)

# Define a route handler for the root path ('/')
@app.route('/')
def home():
    """
    This function renders the home page template ("index.html")
    and passes the list of movie names ("movies_list") as a context variable.
    """

    # Commented out line: Assigning a list of colors (not relevant to the current functionality)
    # colours = ['Red', 'Blue', 'Black', 'Orange']

    # Pass the list of movie names to the template for potential use (e.g., displaying a dropdown menu)
    return render_template("index.html", movies_list=movies_list)


@app.route('/guest', methods=["POST"])
def Guest():
    """
    This function handles POST requests to the '/guest' route.
    It retrieves movie name and top N preference from the form data,
    fetches recommendations, saves them to the database, and displays them as an HTML table.
    """

    if request.method == 'POST':
        # Extract movie name and top N preference from the form data
        mn = request.form["mn"]
        tp = request.form["tp"]

        # Convert top N preference to integer
        top_n = get_recommendations(mn, topN=int(tp))

        # Save recommendations to the database (top_10 table)
        top_n.to_sql('top_10', con=engine, if_exists='replace', chunksize=1000, index=False)

        # Convert recommendations DataFrame to HTML table with styling
        html_table = top_n.to_html(classes='table table-striped')
        
        # Define custom CSS styling for the HTML table (within the returned string)
        return render_template("data.html", Y="Results have been saved in your database", Z=f"""
            <style>
                .table {{
                    width: 50%;
                    margin: 0 auto;
                    border-collapse: collapse; 
                }}
                .table thead {{
                    background-color: #39648f;
                }}
                .table th, .table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }}
                .table td {{
                    background-color: #5e617d;
                }}
                .table tbody th {{
                    background-color: #ab2c3f;
                }}
            </style>
            {html_table}
        """)

# Check if the script is run directly (not imported as a module)
if __name__ == '__main__':
    # Run the Flask development server in debug mode
    app.run(debug=False)

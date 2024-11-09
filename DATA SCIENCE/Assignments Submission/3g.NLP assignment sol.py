import pandas as pd  # Pandas library for data manipulation and analysis.
import re  # Regular expression library for pattern matching.
import nltk  # Natural Language Toolkit library for text processing.
import matplotlib.pyplot as plt  # Library for plotting data.
from collections import Counter  # Tool for counting occurrences of items.
from nltk.stem import PorterStemmer, WordNetLemmatizer  # Tools for reducing words to root forms.
from nltk.tokenize import word_tokenize  # Tokenizer for splitting text into words.
from sklearn.feature_extraction.text import CountVectorizer  # For text vectorization (converting text to numerical data).
from sklearn.decomposition import LatentDirichletAllocation  # Tool for LDA topic modeling.
from gensim.summarization import summarize
# Load dataset
data = pd.read_csv('Data.csv')  # Load data from 'Data.csv' file into a DataFrame.

# Inspect the data to find the name of the text column
print(data.head())  # Display the first few rows to understand the structure and locate the text column.


# Define a function for basic text cleaning
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters (punctuation).
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space.
    text = re.sub(r'\d', '', text)    # Remove all digits.
    return text.lower()               # Convert text to lowercase.
# Apply cleaning function to the text column
data['cleaned_text'] = data['text'].apply(clean_text)  # Apply `clean_text` function to the 'text' column.

# Initialize vectorizer with English stop words removed
vectorizer = CountVectorizer(stop_words='english')  # Initialize vectorizer to remove English stop words.
X = vectorizer.fit_transform(data['cleaned_text'])  # Apply vectorizer to `cleaned_text` to create word count vectors.

# Check dimensions of the transformed data
print("Shape of word count vector:", X.shape)  # Print dimensions of the vectorized text data.
# Download NLTK resources
nltk.download('punkt')  # Download tokenizer data.
nltk.download('wordnet')  # Download WordNet lemmatizer data.

stemmer = PorterStemmer()  # Initialize a stemmer.
lemmatizer = WordNetLemmatizer()  # Initialize a lemmatizer.
# Function to apply stemming and lemmatization
def stem_and_lemmatize(text):
    tokens = word_tokenize(text)  # Split text into words.
    stems = [stemmer.stem(word) for word in tokens]  # Apply stemming.
    lemmas = [lemmatizer.lemmatize(word) for word in stems]  # Apply lemmatization to stemmed words.
    return ' '.join(lemmas)  # Join words back into a single string.
# Apply to the cleaned text
data['processed_text'] = data['cleaned_text'].apply(stem_and_lemmatize)  # Apply to create `processed_text` column.
# Combine all words into a single list
all_words = ' '.join(data['processed_text']).split()  # Create a list of all words in `processed_text`.
word_counts = Counter(all_words)  # Count occurrences of each word.

# Display the top 10 most common words
common_words = word_counts.most_common(10)  # Get the 10 most common words.
print("Most common words:", common_words)  # Print common words and their counts.

# Bar plot of the top 10 words
words, counts = zip(*common_words)  # Unpack words and counts.
plt.bar(words, counts)  # Create a bar plot of word frequencies.
plt.title("Top 10 Most Common Words")  # Add a title.
plt.show()  # Display the plot.
# Define LDA model with 5 topics (adjust the number of topics as needed)
lda = LatentDirichletAllocation(n_components=5, random_state=0)  # Initialize LDA with 5 topics.
lda.fit(X)  # Fit the LDA model to the word count vectors.
# Function to display LDA topics
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):  # Loop through each topic.
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))  # Print top words in the topic.
# Display LDA topics
display_topics(lda, vectorizer.get_feature_names_out(), 10)  # Show top 10 words for each topic.
# Summarize each document
data['summary'] = data['processed_text'].apply(lambda x: summarize(x, word_count=50))  # Adjust word count as needed
# Check a few summaries
print(data[['processed_text', 'summary']].head())

import pandas as pd  # Pandas library for data manipulation and analysis.
from sklearn.feature_extraction.text import CountVectorizer  # For converting text data into a word count matrix.
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD  # For LDA and LSA topic modeling.
# Step 1: Load and Clean the Data
# Specify encoding to handle special characters
with open('NLP-TM.txt', 'r', encoding='utf-8', errors='replace') as file:
    text_data = file.read()  # Read the entire content of the file into a single string.
# Step 1: Load and Clean the Data
# Specify encoding to handle special characters
with open('NLP-TM.txt', 'r', encoding='utf-8', errors='replace') as file:
    text_data = file.read()  # Read the entire content of the file into a single string.
# Basic Cleaning Function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters (e.g., punctuation).
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace characters with a single space.
    return text.lower()  # Convert text to lowercase for uniformity.
cleaned_text = clean_text(text_data)  # Apply the `clean_text` function to the entire text data.
cleaned_text = clean_text(text_data)  # Apply the `clean_text` function to the entire text data.
# Vectorize the Text
vectorizer = CountVectorizer(stop_words='english')  # Initialize the vectorizer to ignore English stop words.
X = vectorizer.fit_transform([cleaned_text])  # Convert the cleaned text into a sparse word count matrix.
# LDA Topic Modeling
lda = LatentDirichletAllocation(n_components=5, random_state=0)  # Initialize LDA model with 5 topics.
lda.fit(X)  # Fit the LDA model to the word count matrix.
# Function to Display Topics
def display_topics(model, feature_names, num_top_words):
    topics = {}  # Dictionary to store topics and their words.
    for idx, topic in enumerate(model.components_):  # Iterate over each topic.
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]  # Select top words.
        topics[f'Topic {idx + 1}'] = topic_words  # Store top words for each topic.
    return topics
lda_topics = display_topics(lda, vectorizer.get_feature_names_out(), 10)  # Get top 10 words for each LDA topic.
# LSA Topic Modeling
lsa = TruncatedSVD(n_components=5, random_state=0)  # Initialize LSA with 5 components.
lsa.fit(X)  # Fit the LSA model to the word count matrix.
lsa_topics = display_topics(lsa, vectorizer.get_feature_names_out(), 10)  # Get top 10 words for each LSA topic.
try:
    summary = summarize(cleaned_text, ratio=0.1)  # Generate summary at 10% of original text length.
except ValueError as e:
    summary = "Summarization failed due to insufficient length of processed text."  # Handle errors if text is too short.
# Display Topics and Summary
print("LDA Topics:", lda_topics)  # Print the LDA topics.
print("LSA Topics:", lsa_topics)  # Print the LSA topics.
print("Summary:", summary)  # Print the text summary.


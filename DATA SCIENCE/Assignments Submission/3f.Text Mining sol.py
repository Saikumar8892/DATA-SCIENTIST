#---------------TEXT MINING-----------------------------
#Text mining, also known as text data mining, is the process of extracting useful information from unstructured text data. It involves techniques from statistics, machine learning, and natural language processing (NLP) to analyze text, identify patterns, and derive insights. Here’s a brief line-by-line overview of the tasks you outlined:
#Extract Content:Gather text from twenty newspaper articles on any trending topic in India to analyze public opinion on current issues.
#Perform Sentiment Analysis:Use sentiment analysis to assess the emotional tone (positive, negative, or neutral) of the extracted articles, determining public sentiment toward the topic.
#Unigram and Bigram Word Clouds:Build word clouds that highlight frequently used single words (unigrams) and two-word phrases (bigrams) to visually represent common terms and themes in the articles.
#Deployment:Deploy the analysis results, making the insights accessible to end-users or stakeholders through a web application or other platform.
#Step 1: Extract Content of Newspaper Articles
#To gather articles, we’ll use newspaper3k, which simplifies news extraction by allowing us to directly download and parse the article’s content.
#Install newspaper3k
#pip install newspaper3k
from newspaper import Article
import nltk
#pip install lxml_html_clean
nltk.download('punkt')  # Download Punkt tokenizer needed for sentence tokenization

# List of URLs of trending news articles (Replace with current URLs)
urls = [
    "https://www.ndtv.com/india-news/latest-news",
    "https://www.thehindu.com/news/national/",
    "https://indianexpress.com/section/india/",
    "https://timesofindia.indiatimes.com/india",
    "https://www.livemint.com/news/india",
    "https://www.hindustantimes.com/india-news/",
    "https://www.businesstoday.in/latest/economy",
    "https://www.moneycontrol.com/news/india/",
    "https://www.theweek.in/news/india.html",
    "https://www.firstpost.com/category/india",
    "https://www.bbc.com/news/world/asia/india",
    "https://scroll.in/latest",
    "https://www.republicworld.com/india-news/",
    "https://www.cnbctv18.com/india/",
    "https://www.financialexpress.com/india-news/",
    "https://www.news18.com/india/",
    "https://www.deccanherald.com/national",
    "https://www.indiatoday.in/india",
    "https://www.dnaindia.com/india",
    "https://www.outlookindia.com/national" 
]
# Initialize an empty list to store article contents
articles_content = []
for url in urls:
    try:
        article = Article(url)  # Initialize the article object with the URL
        article.download()      # Download the article
        article.parse()         # Parse the article text
        articles_content.append(article.text)  # Append article content to the list
        print(f"Successfully downloaded article from {url}")
    except Exception as e:
        print(f"Failed to download article from {url}: {e}")
#Explanation:
#Article(url): Initializes an Article object with the URL of the article.
#article.download(): Downloads the content from the URL.
#article.parse(): Parses the article content to extract text.
#articles.append(article.text): Appends the parsed text to a list.
#Perform Sentiment Analysis
#We’ll use TextBlob to analyze sentiment for each article’s content.
#Install TextBlob
#pip install textblob
#Performing Sentiment Analysis:
from textblob import TextBlob
sentiments = []  # List to store sentiment scores
for content in articles_content:
    blob = TextBlob(content)      # Initialize a TextBlob object for each article
    sentiments.append(blob.sentiment.polarity)  # Get polarity (-1 to 1) and store it
#Build Unigram and Bigram Word Clouds
#pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
# Join all articles into a single text
text = " ".join(articles_content)
# Check if articles contain text
print("Number of articles:", len(articles_content))
for i, article_text in enumerate(articles_content, 1):
    print(f"Article {i} content: {article_text[:100]}...")  # Print first 100 characters of each article
# Unigram Word Cloud
wordcloud_unigram = WordCloud(width=800, height=400, background_color='white').generate(article_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_unigram, interpolation='bilinear')
plt.axis('off')
plt.title('Unigram Word Cloud')
plt.show()
# Bigram Word Cloud
vectorizer = CountVectorizer(ngram_range=(2, 2))  # Initialize for bigrams
bigrams = vectorizer.fit_transform([article_text])        # Generate bigram counts
bigram_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), bigrams.toarray().sum(axis=0))))
wordcloud_bigram = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigram_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_bigram, interpolation='bilinear')
plt.axis('off')
plt.title('Bigram Word Cloud')
plt.show()
#Explanation:
#WordCloud(...).generate(text): Creates a word cloud from text.
#CountVectorizer(ngram_range=(2, 2)): Configures CountVectorizer to extract bigrams (word pairs).
#generate_from_frequencies(bigram_counts): Creates a word cloud from bigram frequencies.
#To create a basic web app, we can use Flask.
#pip install Flask
from flask import Flask, render_template, jsonify
import json
app = Flask(__name__)
@app.route('/')
def home():
    return jsonify(sentiments=sentiments)
if __name__ == '__main__':
    app.run(debug=False)
#Explanation:
#Flask(__name__): Initializes the Flask app.
#@app.route('/'): Defines a route for the home page.
#jsonify(sentiments=sentiments): Sends JSON data (sentiments) to the browser.
#app.run(debug=True): Runs the app in debug mode for easy testing.

#-------------Task-2------------------------------
#-----------------Extract IMDB Movie Reviews and Perform Sentiment Analysis--------------------
#Extract Reviews from IMDB
#Using BeautifulSoup to scrape reviews.
#Install BeautifulSoup and requests:
#Code to Scrape IMDB Reviews:
import requests
from bs4 import BeautifulSoup
url = "https://www.imdb.com/title/tt4154796/reviews"  # Replace with the specific movie's reviews page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
reviews = []
for review in soup.find_all("div", {"class": "text show-more__control"}):  # Update class based on IMDB structure
    reviews.append(review.text)
#Explanation:
#requests.get(url): Sends a GET request to fetch the webpage content.
#BeautifulSoup(response.text, 'html.parser'): Parses HTML content.
#find_all("div", {"class": "text show-more__control"}): Finds all review elements.
#Sentiment Analysis on Reviews
#Using TextBlob for sentiment analysis.
review_sentiments = [TextBlob(review).sentiment.polarity for review in reviews]
#Explanation:
#[TextBlob(review).sentiment.polarity for review in reviews]: Calculates sentiment polarity for each review.
# pip install newspaper3k
'''
Features of Newspaper3k

Multi-threaded article download framework
News URL identification
Text extraction from HTML
Top image extraction from HTML
All image extraction from HTML
Keyword extraction from text
Summary extraction from text
Author extraction from text
Google trending terms extraction
Works in 10+ languages (English, Chinese, German, Arabic,…)
'''


# import pandas as pd
import matplotlib.pyplot as plt  # Importing the matplotlib library for visualization
from wordcloud import WordCloud  # Importing the WordCloud class for generating word clouds
import re  # Importing the re module for regular expressions

import newspaper  # Importing the newspaper library for web scraping
from newspaper import Article  # Importing the Article class for article extraction

# Checking the supported languages for newspaper library
newspaper.languages()  # Printing the supported languages by the newspaper library

# Documentation of the newspaper3k module 
# Newspaper is an amazing python library for extracting & curating articles
# https://newspaper.readthedocs.io/en/latest/
# https://pypi.org/project/newspaper3k/

# Data Extraction from Times of India e-portal
url = 'https://timesofindia.indiatimes.com/sports/cricket/india-in-west-indies/ravichandran-ashwin-indias-greatest-match-winner-since-anil-kumble/articleshow/101793774.cms'

# https://timesofindia.indiatimes.com/world/rest-of-world/singapore-hangs-indian-origin-man-over-1-kg-of-cannabis/articleshow/99800442.cms
# https://economictimes.indiatimes.com/markets/stocks/news/facebook-owner-meta-touts-ai-might-as-digital-ads-boost-outlook-shares-jump/articleshow/99801441.cms
# BBC: https://www.bbc.com/news/technology-65410293


# If no language is specified, Newspaper library will attempt to auto-detect a language.

# Creating an Article object with the specified URL and language settings
article_name = Article(url, language="en")

# Downloading the content of the article from the URL
article_name.download()

# Parsing the content from the HTML document
article_name.parse()

# Extracting the HTML content of the article
article_name.html

# Applying natural language processing (NLP) to extract keywords, summary, etc.
article_name.nlp()

# Printing the title of the article
print("Article Title:")
print(article_name.title)  # Printing the title of the article
print("\n")

print("Article Text:") 
print(article_name.text) # prints the entire text of the article
print("\n") 

print("Article Summary:") 
print(article_name.summary) # prints the summary of the article
print("\n") 

print("Article Keywords:")
print(article_name.keywords) # prints the keywords of the article


# Opening a text file named "News1.txt" in write mode ('w+')
file1 = open("News1.txt", "w+")

# Writing the title of the article into the text file
file1.write("Title:\n")
file1.write(article_name.title)

# Writing the article text into the text file
file1.write("\n\nArticle Text:\n")
file1.write(article_name.text)

# Writing the article summary into the text file
file1.write("\n\nArticle Summary:\n")
file1.write(article_name.summary)

# Writing the article keywords into the text file
file1.write("\n\n\nArticle Keywords:\n")
keywords = '\n'.join(article_name.keywords)  # Joining the keywords list into a string with newline separators
file1.write(keywords)

# Closing the text file
file1.close()

# Opening the file "News1.txt" in read mode and reading its content
with open("News1.txt", "r") as file2:
    text = file2.read()

# Removing non-alphabetic characters and converting the text to lowercase using regular expressions
TOInews = re.sub("[^A-Za-z" "]+", " ", text).lower()

# Tokenizing the cleaned text
TOInews_tokens = TOInews.split(" ")

# Opening the file containing stop words and reading its content
with open("D:\\Data\\textmining\\stop.txt", "r") as sw:
    stop_words = sw.read()

# Splitting the stop words into a list using newline separators
stop_words = stop_words.split("\n")

# Filtering out tokens that are not in the stop words list
tokens = [w for w in TOInews_tokens if not w in stop_words]

# Importing Counter from collections module to count token frequencies
from collections import Counter

# Counting frequencies of each token
tokens_frequencies = Counter(tokens)

# Sorting tokens frequencies in descending order
tokens_frequencies = sorted(tokens_frequencies.items(), key=lambda x: x[1])

# Reversing the sorted tokens frequencies to get them in descending order
frequencies = list(reversed([i[1] for i in tokens_frequencies]))
words = list(reversed([i[0] for i in tokens_frequencies]))

# Barplot of top 10 
# import matplotlib.pyplot as plt

# Creating a bar plot using matplotlib's bar function
plt.bar(height=frequencies[0:11],  # Heights of the bars, representing token frequencies
        x=list(range(0, 11)),  # X-coordinates of the bars (0 to 10)
        color=['red', 'green', 'black', 'yellow', 'blue', 'pink', 'violet'])  # Colors for the bars

# Setting custom x-tick labels for the bar plot
plt.xticks(list(range(0, 11)), words[0:11])  # Setting tokens as x-tick labels for the first 11 tokens
plt.xlabel("Tokens")  # Setting the label for the x-axis
plt.ylabel("Count")   # Setting the label for the y-axis
plt.show()  # Displaying the bar plot
##########


# Joinining all the tokens into single paragraph 
# Joining the words into a single string separated by spaces
cleanstrng = " ".join(words)

# Generating a word cloud using WordCloud object
wordcloud_ip = WordCloud(background_color='White',  # Setting background color to white
                         width=2800, height=2400)    # Setting width and height of the word cloud

# Generating the word cloud image from the cleaned string
wordcloud_ip = wordcloud_ip.generate(cleanstrng)

# Turning off axis display
plt.axis("off")

# Displaying the word cloud image
plt.imshow(wordcloud_ip)


# positive words
# Opening the file containing positive words in read mode and reading its content
with open("D:\\Data\\textmining\\positive-words.txt", "r") as pos:
    poswords = pos.read().split("\n")

# Extracting positive tokens from the list of words
pos_tokens = " ".join([w for w in words if w in poswords])

# Generating a word cloud for positive words using WordCloud object
wordcloud_positive = WordCloud(background_color='White',  # Setting background color to white
                               width=1800, height=1400)    # Setting width and height of the word cloud

# Generating the word cloud image from positive tokens
wordcloud_positive = wordcloud_positive.generate(pos_tokens)

# Creating a new figure for the positive word cloud
plt.figure(2)

# Turning off axis display
plt.axis("off")

# Displaying the positive word cloud image
plt.imshow(wordcloud_positive)


# Negative words
# Opening the file containing negative words in read mode and reading its content
with open("D:\\Data\\textmining\\negative-words.txt", "r") as neg:
    negwords = neg.read().split("\n")

# Extracting negative tokens from the list of words
neg_tokens = " ".join([w for w in words if w in negwords])

# Generating a word cloud for negative words using WordCloud object
wordcloud_negative = WordCloud(background_color='black',  # Setting background color to black
                               width=1800, height=1400)    # Setting width and height of the word cloud

# Generating the word cloud image from negative tokens
wordcloud_negative = wordcloud_negative.generate(neg_tokens)

# Creating a new figure for the negative word cloud
plt.figure(3)

# Turning off axis display
plt.axis("off")

# Displaying the negative word cloud image
plt.imshow(wordcloud_negative)


'''Bi-gram Wordcloud'''
# Downloading necessary NLTK data (if not already downloaded)
import nltk
nltk.download('punkt')

# Generating bigrams from the list of words
bigrams_list = list(nltk.bigrams(words))

# Creating a list of bigrams where each bigram is represented as a string with two words separated by a space
dictionary2 = [' '.join(tup) for tup in bigrams_list]

# Using count vectorizer to view the frequency of bigrams
# Importing necessary module
from sklearn.feature_extraction.text import CountVectorizer

# Creating a CountVectorizer object with ngram range set to (2, 2) for extracting bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))

# Transforming the list of bigram strings into a bag of words representation
bag_of_words = vectorizer.fit_transform(dictionary2)

# Extracting vocabulary (bigrams) and their corresponding indices
vocabulary = vectorizer.vocabulary_

# Summing up the occurrences of each bigram across all documents
sum_words = bag_of_words.sum(axis=0)

# Creating a list of tuples containing bigrams and their frequencies
words_freq = [(word, sum_words[0, idx]) for word, idx in vocabulary.items()]

# Sorting the list of bigrams based on their frequencies in descending order
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Creating a dictionary containing the top 100 most frequent bigrams
words_dict = dict(words_freq[:100])

# Creating a WordCloud object for displaying the top 100 most frequent bigrams
wordcloud_2 = WordCloud(background_color='white', width=1800, height=1400)

# Creating a new figure for the bigram word cloud
plt.figure(4)

# Generating the word cloud image from the bigram frequencies
wordcloud_2.generate_from_frequencies(words_dict)

# Displaying the bigram word cloud image
plt.imshow(wordcloud_2)



''' Emotion Mining'''
# pip install text2emotion
# Importing necessary libraries
import text2emotion as te
import pandas as pd

# Defining the input text
text = "I was asked to sign a third party contract a week out from stay. If it wasn't an 8 person group that took a lot of wrangling I would have cancelled the booking straight away. Bathrooms - there are no stand alone bathrooms. Please consider this - you have to clear out the main bedroom to use that bathroom. Other option is you walk through a different bedroom to get to its en-suite. Signs all over the apartment - there are signs everywhere - some helpful - some telling you rules. Perhaps some people like this but It negatively affected our enjoyment of the accommodation. Stairs - lots of them - some had slightly bending wood which caused a minor injury."

# Getting emotions from the input text
emotions_text = te.get_emotion(text)

# Displaying the emotions extracted from the text
emotions_text

# Capturing the emotions from the token 'work'
emotion_work = te.get_emotion('work')
emotion_work

# Capturing the emotions from the token 'worst'
emotion_worst = te.get_emotion('worst')
emotion_worst

# Capturing the emotions from the token 'proper'
emotion_proper = te.get_emotion('proper')
emotion_proper


# List to store emotions for each token
emosions = []

# Iterate over each token and capture its emotions
for i in words:
    # Get emotions for the current token
    emosions_r = te.get_emotion(i)
    # Append emotions to the list
    emosions.append(emosions_r)

# Convert the list of emotions into a DataFrame
emosions_df = pd.DataFrame(emosions)

# Create a DataFrame for tokens
tokens_df = pd.DataFrame(tokens, columns=['words'])

# Concatenate tokens DataFrame and emotions DataFrame along the columns
emp_emotions = pd.concat([tokens_df, emosions_df], axis=1)

# Plot the sum of each emotion across all tokens
emp_emotions[['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']].sum().plot.bar()


########## End ###########
# Alternately code for wordcloud.
# Create a word cloud
# Generate a word cloud from the text
wordcloud = WordCloud().generate(text)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')

# Turn off axis
plt.axis("off")

# Show the plot
plt.show()

##########
# Specify the URL of the article to be scraped
url = 'https://www.marketwatch.com/story/paypal-sees-record-earnings-volume-amid-sustained-e-commerce-surge-11620246188?siteid=yhoof2'

# Create an Article object and specify the language as English
article = Article(url, language="en")

# Download the article content
article.download()

# Parse the downloaded content
article.parse()

# Perform natural language processing on the article
article.nlp()

# Open a file in write mode
file1 =  open("article51.txt", "w+")

# Write the article title to the file
file1.write(article.title)

# Write a new line
file1.write("\n\n")

# Write the article text to the file
file1.write(article.text)

# Close the file
file1.close()

################

# pip install newsapi-python
# from newsapi import NewsApiClient
# import pandas as pd
# import datetime as dt

# newsapi = NewsApiClient(api_key = 'f860364762db4c5a961ca7cc8765f539')

# data = newsapi.get_everything(q = 'illegal drugs', language = 'en', page_size = 100)

# articles = data['articles']

# df = pd.DataFrame(articles)

# df

# df.drop('publishedAt', axis = 1, inplace = True)

# df.to_csv('Article.csv')

# a = list(df['content'])
  
# # converting list into string and then joining it with space
# b = ' '.join(str(e) for e in a)
  
# with open("Article100.txt", "w", encoding = 'utf8') as output:
#     output.write(str(b))

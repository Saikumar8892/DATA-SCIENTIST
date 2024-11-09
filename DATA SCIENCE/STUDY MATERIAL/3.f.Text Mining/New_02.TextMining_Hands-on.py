
# Tokenization

import re  # Importing the regular expression module

sentence5 = 'Sharat tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official \
@indian_army #India #70thRepublic_Day. For more photos ping me sharat@photoking.com :)"'

sentence5.split()  # Splitting the sentence into words using whitespace as the delimiter

# Using regular expression to substitute any non-alphanumeric characters (except whitespace) with a whitespace,
# and then splitting the sentence into words
re.sub(r'([^\s\w]|_)+', ' ', sentence5).split()


# Extracting n-grams
# n-grams can be extracted from 3 different techniques:
# listed below are:
# 1. Custom defined function
# 2. NLTK
# 3. TextBlob

# Extracting n-grams using customed defined function
import re  # Importing the regular expression module

# Defining a function to extract n-grams from an input string
def n_gram_extractor(input_str, n):
    # Using regular expression to substitute any non-alphanumeric characters (except whitespace) with a whitespace,
    # and then splitting the input string into tokens (words)
    tokens = re.sub(r'([^\s\w]|_)+', ' ', input_str).split()
    
    # Iterating over the tokens to extract n-grams
    for i in range(len(tokens)-n+1):
        # Printing the n-gram extracted from the tokens
        print(tokens[i:i+n])

# Extracting bigrams (2-grams) from the given input string
n_gram_extractor('The cute little boy is playing with the kitten.', 2)

# Extracting trigrams (3-grams) from the given input string
n_gram_extractor('The cute little boy is playing with the kitten.', 3)


# Extracting n-grams with nltk
from nltk import ngrams  # Importing the ngrams function from NLTK

# Generating and printing bigrams (2-grams) from the given input string
list(ngrams('The cute little boy is playing with the kitten.'.split(), 2))

# Generating and printing trigrams (3-grams) from the given input string
list(ngrams('The cute little boy is playing with the kitten.'.split(), 3))


# Extracting n-grams using TextBlob
# TextBlob is a Python library for processing textual data.

# pip install textblob

from textblob import TextBlob  # Importing the TextBlob class from the textblob library

# Creating a TextBlob object with the given text
blob = TextBlob("The cute little boy is playing with the kitten.")

# Generating and printing bigrams (2-grams) using TextBlob
blob.ngrams(n=2)

# Generating and printing trigrams (3-grams) using TextBlob
blob.ngrams(n=3)

# Tokenizing texts with different packages: Keras, Textblob
sentence5 = 'Sharat tweeted, "Witnessing 70th Republic Day of India from Rajpath, New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sharat@photoking.com :)"'

# No specific code provided for tokenization with Keras, so commenting isn't possible.

# pip install tensorflow
# pip install keras

# Tokenization with Keras
from keras.preprocessing.text import text_to_word_sequence  # Importing the text_to_word_sequence function from Keras

# Tokenizing the given sentence using Keras text_to_word_sequence
text_to_word_sequence(sentence5)

# Tokenization with TextBlob
from textblob import TextBlob  # Importing the TextBlob class from the textblob library

blob = TextBlob(sentence5)  # Creating a TextBlob object from the given sentence

blob.words  # Accessing the words property of the TextBlob object to tokenize the sentence

# Tokenize sentences using other nltk tokenizers:
# 1. Tweet Tokenizer
# 2. MWE Tokenizer (Multi-Word Expression)
# 3. Regexp Tokenizer
# 4. Whitespace Tokenizer
# 5. Word Punct Tokenizer


# 1. Tweet tokenizer
from nltk.tokenize import TweetTokenizer  # Importing the TweetTokenizer class from NLTK

tweet_tokenizer = TweetTokenizer()  # Initializing a TweetTokenizer object

tweet_tokenizer.tokenize(sentence5)  # Tokenizing the given sentence using the TweetTokenizer

# 2. MWE Tokenizer (Multi-Word Expression)
from nltk.tokenize import MWETokenizer  # Importing the MWETokenizer class from NLTK

# Initializing a MWETokenizer object with a set of words to be treated as one entity
mwe_tokenizer = MWETokenizer([('Republic', 'Day')])

# Adding more words to the set of words to be treated as one entity
mwe_tokenizer.add_mwe(('Indian', 'Army'))

# Tokenizing the sentence using the MWETokenizer. In this case, 'Indian Army' should be treated as a single token,
# but 'Army!' is treated as a separate token because it includes punctuation.
mwe_tokenizer.tokenize(sentence5.split())

# Tokenizing the modified sentence (with punctuation removed) using the MWETokenizer.
# Now, 'Indian Army' will be treated as a single token, as 'Army!' is treated as 'Army'.
mwe_tokenizer.tokenize(sentence5.replace('!', '').split())


# 3. Regexp Tokenizer
from nltk.tokenize import RegexpTokenizer  # Importing the RegexpTokenizer class from NLTK

# Initializing a RegexpTokenizer object with a regular expression pattern
# '\w+' matches any word character (equivalent to [a-zA-Z0-9_])
# '|\$[\d\.]+' matches dollar amounts (e.g., $10.99)
# '|\S+' matches any non-whitespace character
reg_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

# Tokenizing the sentence using the RegexpTokenizer
reg_tokenizer.tokenize(sentence5)


# 4. Whitespace Tokenizer
from nltk.tokenize import WhitespaceTokenizer  # Importing the WhitespaceTokenizer class from NLTK

wh_tokenizer = WhitespaceTokenizer()  # Initializing a WhitespaceTokenizer object

wh_tokenizer.tokenize(sentence5)  # Tokenizing the sentence using the WhitespaceTokenizer


# 5. WordPunct Tokenizer
from nltk.tokenize import WordPunctTokenizer  # Importing the WordPunctTokenizer class from NLTK

sentence6 = sentence5.replace('th', '')  # Creating a modified sentence by removing 'th'

wp_tokenizer = WordPunctTokenizer()  # Initializing a WordPunctTokenizer object

wp_tokenizer.tokenize(sentence5)  # Tokenizing the original sentence using the WordPunctTokenizer

wp_tokenizer.tokenize(sentence6)  # Tokenizing the modified sentence using the WordPunctTokenizer

# Stemming
# Regexp Stemmer
sentence6 = "I love playing Cricket. Cricket players practice hard in their inning"  # Defining a sentence

from nltk.stem import RegexpStemmer  # Importing the RegexpStemmer class from NLTK

regex_stemmer = RegexpStemmer('ing$')  # Initializing a RegexpStemmer object with a regular expression pattern 'ing$'

# Applying stemming to each word in the sentence using the RegexpStemmer and joining the stemmed words into a single string
' '.join([regex_stemmer.stem(wd) for wd in sentence6.split()])


# Porter Stemmer
sentence7 = "Before eating, it would be nice to sanitize your hands with a sanitizer"  # Defining a sentence

from nltk.stem.porter import PorterStemmer  # Importing the PorterStemmer class from NLTK

ps_stemmer = PorterStemmer()  # Initializing a PorterStemmer object

# Applying stemming to each word in the sentence using the PorterStemmer and joining the stemmed words into a single string
' '.join([ps_stemmer.stem(wd) for wd in sentence7.split()])


# Lemmatization
# pip install -U earthy
from earthy.nltk_wrappers import lemmatize_sent  # Importing the lemmatize_sent function from the earthy.nltk_wrappers module

sentence8 = "The codes executed today are far better than what we execute generally."  # Defining a sentence

lemmatize_sent(sentence8)  # Lemmatizing the given sentence using the lemmatize_sent function

# Unpacking the results of lemmatization into separate lists for words, lemmas, and tags
words, lemmas, tags = zip(*lemmatize_sent(sentence8))

lemmas  # Displaying the lemmas extracted from the sentence


# Singularize & Pluralize words
from textblob import TextBlob  # Importing the TextBlob class from the textblob library

sentence9 = TextBlob('She sells seashells on the seashore')  # Creating a TextBlob object with the given sentence

sentence9.words  # Accessing the words property of the TextBlob object to tokenize the sentence into words

sentence9.words[2].singularize()  # Singularizing the third word in the sentence ('seashells')

sentence9.words[5].pluralize()  # Pluralizing the sixth word in the sentence ('seashore')


# Language Translation
# From Spanish to English

from textblob import TextBlob  # Importing the TextBlob class from the textblob library

en_blob = TextBlob(u'muy bien')  # Creating a TextBlob object with the Spanish text 'muy bien'

# Translating the Spanish text to English, specifying 'es' as the source language and 'en' as the target language
en_blob.translate(from_lang='es', to='en') 


# Custom Stop words removal
from nltk import word_tokenize  # Importing the word_tokenize function from NLTK

sentence9 = "She sells seashells on the seashore"  # Defining a sentence

custom_stop_word_list = ['she', 'on', 'the', 'am', 'is', 'not']  # Defining a custom list of stop words

# Filtering out stop words from the sentence using list comprehension:
# - Tokenizing the sentence into words using word_tokenize
# - Checking if the lowercase version of each word is not in the custom_stop_word_list
# - Joining the remaining words into a single string
' '.join([word for word in word_tokenize(sentence9) if word.lower() not in custom_stop_word_list])


# Extracting general features from raw texts

# Number of words
# Detect presence of wh words
# Polarity
# Subjectivity
# Language identification

import pandas as pd  # Importing the pandas library

# Creating a DataFrame with the given list of sentences as data and setting the column name as 'text'
df = pd.DataFrame([['The vaccine for covid-19 will be announced on 1st August.'],
                   ['Do you know how much expectation the world population is having from this research?'],
                   ['This risk of virus will end on 31st July.']])

# Assigning column name 'text' to the DataFrame
df.columns = ['text']

# Displaying the DataFrame
df

from textblob import TextBlob  # Importing the TextBlob class from the textblob library

# Calculating the number of words in each text and adding the result as a new column 'number_of_words' to the DataFrame
df['number_of_words'] = df['text'].apply(lambda x: len(TextBlob(x).words))

# Displaying the number of words for each text
df['number_of_words']

# Defining a set of wh-words
wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])

# Detecting the presence of wh-words in each text and adding the result as a new column 'are_wh_words_present' to the DataFrame
df['are_wh_words_present'] = df['text'].apply(lambda x: True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False)

# Displaying whether wh-words are present in each text
df['are_wh_words_present']


# Polarity
# Calculating the sentiment polarity for each text and adding the result as a new column 'polarity' to the DataFrame
df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Displaying the sentiment polarity for each text
df['polarity']

# Subjectivity
# Calculating the sentiment subjectivity for each text and adding the result as a new column 'subjectivity' to the DataFrame
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Displaying the sentiment subjectivity for each text
df['subjectivity']

# Language of the sentence
# Language Detector using spacy
# pip install spacy
# pip install spacy_langdetect
# pip install httplib2

# import spacy
# from spacy.language import Language
# from spacy_langdetect import LanguageDetector
# def get_lang_detector(nlp, name):
#     return LanguageDetector()
# nlp = spacy.load("en_core_web_sm")
# Language.factory("language_detector", func=get_lang_detector)
# nlp.add_pipe('language_detector', last=True)
# text = 'This is an english text.'
# text = 'muy bien'
# doc = nlp(text)
# print(doc._.language)


# Bag of Words
import pandas as pd  # Importing the pandas library
from sklearn.feature_extraction.text import CountVectorizer  # Importing the CountVectorizer class from scikit-learn

# Defining a corpus of text documents
corpus = [
    'At least seven Indian pharma companies are working to develop a vaccine against coronavirus',
    'the deadly virus that has already infected more than 14 million globally.',
    'Bharat Biotech, Indian Immunologicals, are among the domestic pharma firms working on the coronavirus vaccines in India.'
]

bag_of_words_model = CountVectorizer()  # Initializing a CountVectorizer object

# Transforming the corpus into a bag-of-words representation using the CountVectorizer and printing the dense matrix
print(bag_of_words_model.fit_transform(corpus).todense())  # bag of words

# Converting the bag-of-words representation into a DataFrame
bag_of_word_df = pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())

# Assigning sorted vocabulary as column names to the DataFrame
bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)

# Displaying the DataFrame
bag_of_word_df.head()

# Creating a bag-of-words model with only the top 5 frequent terms
bag_of_words_model_small = CountVectorizer(max_features=5)  # Initializing a CountVectorizer object with max_features=5

# Transforming the corpus into a bag-of-words representation with only the top 5 frequent terms
bag_of_word_df_small = pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())

# Assigning sorted vocabulary as column names to the DataFrame
bag_of_word_df_small.columns = sorted(bag_of_words_model_small.vocabulary_)

# Displaying the DataFrame
bag_of_word_df_small.head()

# TF-IDF (Term Frequency-Inverse Document Frequency)
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing the TfidfVectorizer class from scikit-learn

tfidf_model = TfidfVectorizer()  # Initializing a TfidfVectorizer object

# Transforming the corpus into a TF-IDF representation
tfidf_matrix = tfidf_model.fit_transform(corpus)

# Printing the TF-IDF matrix in dense format
print(tfidf_matrix.todense())

# Converting the TF-IDF matrix into a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.todense())

# Assigning sorted vocabulary as column names to the DataFrame
tfidf_df.columns = sorted(tfidf_model.vocabulary_)

# Displaying the DataFrame
tfidf_df.head()

# Creating a TF-IDF representation with only the top 5 frequent terms
tfidf_model_small = TfidfVectorizer(max_features=5)  # Initializing a TfidfVectorizer object with max_features=5

# Transforming the corpus into a TF-IDF representation with only the top 5 frequent terms
tfidf_matrix_small = tfidf_model_small.fit_transform(corpus)

# Converting the TF-IDF matrix into a DataFrame
tfidf_df_small = pd.DataFrame(tfidf_matrix_small.todense())

# Assigning sorted vocabulary as column names to the DataFrame
tfidf_df_small.columns = sorted(tfidf_model_small.vocabulary_)

# Displaying the DataFrame
tfidf_df_small.head()


# Feature Engineering (Text Similarity)
from nltk import word_tokenize  # Importing the word_tokenize function from NLTK
# from nltk.stem import WordNetLemmatizer  # Lemmatizer is commented out
from earthy.nltk_wrappers import lemmatize_sent  # Importing the lemmatize_sent function from the earthy.nltk_wrappers module
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing the TfidfVectorizer class from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity  # Importing the cosine_similarity function from scikit-learn

# Defining pairs of sentences for comparison
pair1 = ["Do you have Covid-19", "Your body temperature will tell you"]
pair2 = ["I travelled to Malaysia.", "Where did you travel?"]
pair3 = ["He is a programmer", "Is he not a programmer?"]

def extract_text_similarity_jaccard(text1, text2):
    # Lemmatizing and converting texts to lowercase
    words_text1 = tuple(zip(*lemmatize_sent(text1.lower())))[1]  # Lemmatizing and extracting words from text1
    words_text2 = tuple(zip(*lemmatize_sent(text2.lower())))[1]  # Lemmatizing and extracting words from text2
    
    # Calculating Jaccard similarity
    nr = len(set(words_text1).intersection(set(words_text2)))  # Number of common words
    dr = len(set(words_text1).union(set(words_text2)))  # Number of unique words in both texts combined
    jaccard_sim = nr / dr  # Jaccard similarity calculation
    
    return jaccard_sim

# Computing Jaccard similarity for pair1
extract_text_similarity_jaccard(pair1[0], pair1[1])

# Computing Jaccard similarity for pair2
extract_text_similarity_jaccard(pair2[0], pair2[1])

# Computing Jaccard similarity for pair3
extract_text_similarity_jaccard(pair3[0], pair3[1])

tfidf_model = TfidfVectorizer()  # Initializing a TfidfVectorizer object

# Creating a corpus which will have texts of pair1, pair2 and pair3 respectively
# Combining all texts from pairs into a single list
corpus = [pair1[0], pair1[1], pair2[0], pair2[1], pair3[0], pair3[1]]

# Transforming the corpus into a TF-IDF matrix
tfidf_results = tfidf_model.fit_transform(corpus).todense()
# Note: Here tfidf_results will have tf-idf representation of 
# texts of pair1, pair2 and pair3 in the given order.

# tfidf_results[0], tfidf_results[1] represents pair1
# tfidf_results[2], tfidf_results[3] represents pair2
# tfidf_results[4], tfidf_results[5] represents pair3

#cosine similarity between texts of pair1
cosine_similarity(tfidf_results[0], tfidf_results[1])

#cosine similarity between texts of pair2
cosine_similarity(tfidf_results[2], tfidf_results[3])

#cosine similarity between texts of pair3
cosine_similarity(tfidf_results[4], tfidf_results[5])

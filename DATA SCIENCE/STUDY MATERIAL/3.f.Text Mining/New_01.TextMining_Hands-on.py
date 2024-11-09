# Text Mining and NLP - Hands-on
#################################

sentence = "We are Learning TextMining from 360DigiTMG"  # Assigning a string to the variable 'sentence'

'TextMining' in sentence  # Checking if the substring 'TextMining' is present in the sentence

sentence.index('Learning')  # Finding the index of the substring 'Learning' in the sentence

sentence.split().index('TextMining')  # Splitting the sentence into words and finding the index of 'TextMining'

sentence.split()[2]  # Extracting the third word from the sentence

sentence.split()[2][::-1]  # Reversing the third word in the sentence

words = sentence.split()  # Splitting the sentence into words and storing them in a list named 'words'

first_word = words[0]  # Extracting the first word from the list 'words'

last_word = words[len(words)-1]  # Extracting the last word from the list 'words' using its index

concat_word = first_word + ' ' + last_word  # Concatenating the first and last word with a space in between

print(concat_word)  # Printing the concatenated word

[words[i] for i in range(len(words)) if i%2 == 0]  # Creating a list comprehension to extract words at even indices

sentence[-3:]  # Extracting the last three characters from the sentence using negative indexing

sentence[::-1] # Print entire sentence in reverse order

print(' '.join([word[::-1] for word in words])) # Select each word and print it in reverse


# Word Tokenization 
import nltk  # Importing the Natural Language Toolkit library

nltk.download('punkt')  # Downloading the necessary resource for word tokenization

nltk.download()  # GUI to browse and download additional NLTK resources if needed

from nltk import word_tokenize  # Importing the word_tokenize function from NLTK

words = word_tokenize("I am reading NLP Fundamentals")  # Tokenizing the given sentence into words

print(words)  # Printing the tokenized words

# Parts of Speech Tagging
nltk.download('averaged_perceptron_tagger')  # Downloading the necessary resource for parts of speech tagging

nltk.pos_tag(words)  # Performing parts of speech tagging on the tokenized words


# Stop Words
nltk.download('stopwords')  # Downloading the stopwords corpus from NLTK

from nltk.corpus import stopwords  # Importing the stopwords corpus from NLTK

stop_words = stopwords.words('english')  # Storing the list of English stopwords in 'stop_words'

print(stop_words)  # Printing the list of stopwords

sentence1 = "I am learning NLP. It is one of the most popular library in Python"  # Defining a sentence

sentence_words = word_tokenize(sentence1)  # Tokenizing the sentence into words

print(sentence_words)  # Printing the tokenized words


# Filtering stop words from the input string
sentence_no_stops = ' '.join([word for word in sentence_words if word not in stop_words])  # Filtering out stop words from the tokenized words
print(sentence_no_stops)  # Printing the sentence after removing stop words

# Text Normalization
# Replace words in string
sentence2 = "I visited MY from IND on 14-02-20"  # Defining a sentence with abbreviations

normalized_sentence = sentence2.replace("MY", "Malaysia").replace("IND", "India").replace("-20", "-2020")  # Replacing abbreviations with full forms and normalizing the date format
print(normalized_sentence)  # Printing the normalized sentence


# Spelling Corrections
# pip install autocorrect
from autocorrect import Speller  # Importing the Speller class from the autocorrect library

spell = Speller(lang='en')  # Initializing a spell checker for English language using the Speller class

help(Speller)  # Displaying help information about the Speller class

spell('Natureal')  # Correcting the spelling of the word 'Natureal' using the spell checker

sentence3 = word_tokenize("Ntural Luanguage Processin deals with the art of extracting insightes from Natural Languaes")  # Tokenizing a sentence with intentional typos

print(sentence3)  # Printing the tokenized words of the sentence with typos

sentence_corrected = ' '.join([spell(word) for word in sentence3])  # Correcting the typos in the sentence using the spell checker
print(sentence_corrected)  # Printing the corrected sentence


# Stemming
stemmer = nltk.stem.PorterStemmer()  # Initializing a Porter stemmer from NLTK

stemmer.stem("Programming")  # Stemming the word "Programming" to its root form

stemmer.stem("Programs")  # Stemming the word "Programs" to its root form

stemmer.stem("Jumping")  # Stemming the word "Jumping" to its root form

stemmer.stem("Jumper")  # Stemming the word "Jumper" to its root form

stemmer.stem("battling")  # Stemming the word "battling" to its root form (stemming doesn't consider dictionary words)

stemmer.stem("amazing")  # Stemming the word "amazing" to its root form

# Lemmatization
# Lemmatization looks into dictionary words
import nltk  # Importing the NLTK library
nltk.download('wordnet')  # Downloading WordNet data

from nltk.stem.wordnet import WordNetLemmatizer  # Importing the WordNetLemmatizer class from NLTK

lemmatizer = WordNetLemmatizer()  # Initializing a WordNet lemmatizer

lemmatizer.lemmatize('Programming')  # Lemmatizing the word 'Programming' to its base or dictionary form

lemmatizer.lemmatize('Programs')  # Lemmatizing the word 'Programs' to its base or dictionary form

lemmatizer.lemmatize('battling')  # Lemmatizing the word 'battling' to its base or dictionary form

lemmatizer.lemmatize("amazing")  # Lemmatizing the word 'amazing' to its base or dictionary form


# Named Entity Recognition (NER)
# Chunking (Shallow Parsing) - Identifying named entities
nltk.download('maxent_ne_chunker')  # Downloading the necessary resource for named entity chunking
nltk.download('words')  # Downloading the necessary word list

sentence4 = "We are learning nlp in Python by 360DigiTMG which is based out of India."  # Defining a sentence

# Performing part-of-speech tagging on the tokenized words of the sentence and then performing named entity chunking on the tagged words,
# specifying binary=True to indicate whether named entities should be chunked into individual chunks or not
i = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence4)), binary=True)

# Extracting the named entities from the result of named entity chunking
[a for a in i if len(a)==1]


# Sentence Tokenization
from nltk.tokenize import sent_tokenize  # Importing the sent_tokenize function from NLTK

# Tokenizing the given text into sentences
sent_tokenize("We are learning NLP in Python. Delivered by 360DigiTMG. Do you know where is it located? It is based out of India.")

# WSD
from nltk.wsd import lesk  # Importing the lesk function from NLTK for Word Sense Disambiguation (WSD)

sentence1 = "Keep your savings in the bank"  # Defining a sentence with an ambiguous word 'bank'

# Performing WSD on the tokenized words of the first sentence with the ambiguous word 'bank'
print(lesk(word_tokenize(sentence1), 'bank'))

sentence2 = "It's so risky to drive over the banks of the river"  # Defining another sentence with the ambiguous word 'bank'

# Performing WSD on the tokenized words of the second sentence with the ambiguous word 'bank'
print(lesk(word_tokenize(sentence2), 'bank'))

# "bank" has multiple meanings. 
# The definitions for "bank" can be seen here:

from nltk.corpus import wordnet as wn  # Importing the WordNet corpus from NLTK

# Iterating over all synsets (senses) of the word 'bank' in WordNet and printing each synset along with its definition
for ss in wn.synsets('bank'):
    print(ss, ss.definition())
# synsets (short for Synonym-set) are the groupings of synonymous words
# that express the same concept
'''
#######################################
1.	CC	Coordinating conjunction
2.	CD	Cardinal number
3.	DT	Determiner
4.	EX	Existential there
5.	FW	Foreign word
6.	IN	Preposition or subordinating conjunction
7.	JJ	Adjective
8.	JJR	Adjective, comparative
9.	JJS	Adjective, superlative
10.	LS	List item marker
11.	MD	Modal
12.	NN	Noun, singular or mass
13.	NNS	Noun, plural
14.	NNP	Proper noun, singular
15.	NNPS	Proper noun, plural
16.	PDT	Predeterminer
17.	POS	Possessive ending
18.	PRP	Personal pronoun
19.	PRP$	Possessive pronoun
20.	RB	Adverb
21.	RBR	Adverb, comparative
22.	RBS	Adverb, superlative
23.	RP	Particle
24.	SYM	Symbol
25.	TO	to
26.	UH	Interjection
27.	VB	Verb, base form
28.	VBD	Verb, past tense
29.	VBG	Verb, gerund or present participle
30.	VBN	Verb, past participle
31.	VBP	Verb, non-3rd person singular present
32.	VBZ	Verb, 3rd person singular present
33.	WDT	Wh-determiner
34.	WP	Wh-pronoun
35.	WP$	Possessive wh-pronoun
36.	WRB	Wh-adverb
###################################################
'''
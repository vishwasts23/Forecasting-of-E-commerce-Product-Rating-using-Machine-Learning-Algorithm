from sklearn.model_selection import train_test_split
import sys
from itertools import chain
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import joblib
import spacy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nltk
from pymongo import MongoClient

# nltk.download('wordnet')


# nltk.download('punkt')


# nltk.download('stopwords')
ps = PorterStemmer()

# # # Load the custom CSV file
df = pd.read_csv(
    './GrammarandProductReviews.csv')

# # Define functions for text preprocessing

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(str(text).lower())

    # Remove stop words and punctuation
    filtered_tokens = [
        token for token in tokens if token.isalpha() and token not in stop_words]

    # Stem the tokens
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(
        token) for token in stemmed_tokens]

    # Return the preprocessed text as a string
    return ' '.join(lemmatized_tokens)


def pos_tag_text(text):
    # Tokenize the text
    tokens = (str(text).lower())
    # Perform part-of-speech tagging on the tokens
    pos_tags = pos_tag(tokens)

    # Return the part-of-speech tags as a list of tuples
    return pos_tags


def ner_text(text):
    # Tokenize the text
    tokens = word_tokenize(str(text).lower())

    # Perform named entity recognition on the tokens
    ne_tags = ne_chunk(pos_tag(tokens))

    # Return the named entities as a list of tuples
    return ne_tags


# Preprocess the text reviews in the dataset
df['preprocessed_text'] = df['reviews.text'].apply(preprocess_text)

# # Vectorize the text reviews using a count vectorizer

vectorizer = TfidfVectorizer()
text_vectorized = vectorizer.fit_transform(df['preprocessed_text'])

# # Train a different model on the vectorized text and the ratings
model = LinearRegression()

#model = LogisticRegression()
model = RandomForestRegressor(n_estimators=5)


df['reviews.rating'] = pd.to_numeric(df['reviews.rating'], errors='coerce')
df = df.replace(np.nan, 0, regex=True)

model.fit(text_vectorized, df['reviews.rating'])


joblib.dump(model, "./model2.joblib")
joblib.dump(vectorizer, 'vectorizer2.joblib')


# # pre = model.predict(text_vectorized)


# loaded_rf = joblib.load("./model2.joblib")
# vect = joblib.load("./vectorizer2.joblib")

# # mongoClient = MongoClient()
# # db = mongoClient['products']
# # collection = db['amazon_products']

# # document_list = []
# # documents = list(collection.find(
# #     {}, {"_id": 0, "Product Name": 0, "Price": 0, "overall-ratings": 0}))

# # kk = []
# # for d in documents:
# #     for key, value in d.items():
# #         kk.append(value)

# # # value_list = [list(d.values()) for d in documents]

# # for i in kk:
# #     k = i.split("|")

# #     test_text = k[4]


# kdd = ["the product is good and its quality is excellent"]

# test_text_preprocessed = [preprocess_text(text) for text in kdd]
# test_text_vectorized = vect.transform(test_text_preprocessed)

# predicted_ratings = loaded_rf.predict(test_text_vectorized)

# print(predicted_ratings)


# Split the dataset into a training set and a testing set
# X_train, X_test, y_train, y_test = train_test_split(
#     text_vectorized, df['reviews.rating'], test_size=0.2, random_state=42)

# # Train the model on the training set
# model.fit(X_train, y_train)

# # Evaluate the model's accuracy on the testing set
# accuracy = model.score(X_test, y_test)
# print("Accuracy:", accuracy)

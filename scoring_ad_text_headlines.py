# -*- coding: utf-8 -*-
""" Import Important libraries ✈
"""

import nltk
nltk.download('punkt_tab')
  # Ensure the correct NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

"""
📜 Table of Contents

Introduction
Project Overview
Architecture
Project Structure
Code Implementation
Technologies
Challenges & Insights
Future Roadmap
Conclusion
🌟 Introduction

Use AI to design an algorithm for scoring the  Google Ad Text
Headlines for the website ronspotflexwork.com we

🧩 Project Overview

The keywords are extracted from headlines and keywords.
Tokenize the Headlines
Calculate a Similarity Score

Code for extracting the keywords
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Ensure NLTK resources are downloaded properly
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the headline and keywords
headlines = ["Automate Car Park Now",
"Automate Car Parking",
"Automate Parking Easily",
"Automate Parking Management",
"Automate Parking Now",
"Automate Parking Today",
"Automate Your Parking",
"Automate Your Parking Now",
"Boost Office Parking",
"Efficient Parking Management",
"Manage Office Parking",
"Optimize Office Parking",
"Parking Automation Solution",
"Streamline Office Parking",
"Streamline Parking Today"
]
# Sample keywords
keywords = ["parking lot management app",
    "park on my drive",
     "parking management solutions",
     "automated parking system",
     "parking lot payment systems",
     "parking lot barrier",
     "book parking",
    "parking spot",
    "parking lot management software",
    "secure parking",
    "parking barriers",
    "my car parking space my parking space",
    "automated parking lot management system",
    "car park management"]

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text (tokenization, stopword removal, and lemmatization)
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())  # Convert to lower case for consistency
    # Remove stopwords and non-alphabetic words
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

# Preprocess the headlines
headline_tokens = []
for headline in headlines:  # Iterate through each headline in the list
    headline_tokens.extend(preprocess_text(headline))

# Preprocess the keywords
keyword_tokens = [lemmatizer.lemmatize(word.lower()) for word in keywords if word.lower() not in stop_words]

# Combine both the headline and keyword tokens
all_tokens = headline_tokens + keyword_tokens

# Count the frequency of each word
word_counts = Counter(all_tokens)

# Get the most common words
common_words = word_counts.most_common()

# Display the most common words
print("Most Common Words:", common_words)

"""Computing the scoring of each headline. It is based on:
Relevance Score, Clarity and Readability Score, Conciseness Score,Persuasiveness Score
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import string
import re

# Preprocess text (lowercase, remove punctuation, tokenize)
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)

# Relevance Score (0-4): Cosine Similarity with adjusted weight
vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(keywords + headlines)
def relevance_score(headline):
    headline_vec = vectorizer.transform([headline])
    keyword_vec = vectorizer.transform([" ".join(keywords)])
    score = cosine_similarity(headline_vec, keyword_vec)[0][0]
    return min(int(score * 4), 4)  # Normalize to 0-4 scale

# Clarity & Readability Score (0-2): Adjusted for short headlines
def clarity_score(headline):
    words = preprocess(headline)
    if len(words) < 2:  # Penalize only very short headlines
        return 1
    return 2

# Conciseness Score (0-2): Short headlines get full score
def conciseness_score(headline):
    if len(headline) > 30:
        return 0  # Too long
    return 2  # Ideal for short headlines

# Persuasiveness Score (0-4): Action Words with more weight
def persuasiveness_score(headline):
    action_words = {"automate", "parking", "management", "car", "system", "solution", "app", "drive" "now","today"}
    words = set(preprocess(headline))
    matches = len(action_words.intersection(words))
    return min(matches * 2, 4)  # More weight for action words

# Compute final score (1-10)
def final_score(headline):
    return (relevance_score(headline) + clarity_score(headline) +
            conciseness_score(headline) + persuasiveness_score(headline))

# Score each headline
for h in headlines:
    print(f"{h}: {final_score(h)}/10")

"""AI to generate believable training data for the headlines and keywords

**GPT-2 model to generate multiple ad headlines for automated car parking, printing each one as a separate result**
"""

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
output = generator("Generate an ad headline for automated car parking:", max_length=30, num_return_sequences=10)

for text in output:
    print(text["generated_text"])


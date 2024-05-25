# Step 1: Cleaning data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
from collections import Counter
import math

nltk.download('stopwords')
# import urllib.request

# url = "https://en.wikipedia.org/wiki/Alexander_McQueen"

# # First open, then read the the HTML
# html = urllib.request.urlopen(url).read()

# # Print the HTML code
# print(html)

# # prints the datatype of the output.
# print(type(html))
# from bs4 import BeautifulSoup

# # Parse the HTML using BeautifulSoup
# soup = BeautifulSoup(html, 'html.parser')

# # Extract text from the parsed HTML
# documents = soup.get_text()

# print(documents)
# Function to generate documents
def generate_documents(phrases, num_docs):
    documents = []
    for _ in range(num_docs):
        doc = " ".join(np.random.choice(phrases, np.random.randint(5, 100)))
        documents.append(doc)
    return documents

# Sample phrases
phrases = ["machine learning","computer vision","artificial intelligence"]

# Generate documents
documents = generate_documents(phrases, 3)

def clean_data(document):
    cleaned_doc = ''.join([char.lower() for char in document if char.isalnum() or char.isspace()])
    return cleaned_doc

cleaned_documents = [clean_data(doc) for doc in documents]

# Step 2: Tokenization
tokenized_documents = [word_tokenize(doc) for doc in cleaned_documents]

# Step 3: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_documents = [[lemmatizer.lemmatize(word) for word in doc] for doc in tokenized_documents]

# Step 4: Remove stop words
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
filtered_documents = [[word for word in doc if word not in ENGLISH_STOP_WORDS] for doc in lemmatized_documents]

# Step 5: Get unique words
unique_words = list(set([word for doc in filtered_documents for word in doc]))

# Step 6: Compute TF
def compute_tf(document):
    word_counts = Counter(document)
    total_words = len(document)
    tf = {word: word_counts[word] / total_words for word in word_counts}
    return tf

tf_documents = [compute_tf(doc) for doc in filtered_documents]

# Step 7: Compute IDF
def compute_idf(documents, unique_words):
    idf = {}
    total_documents = len(documents)
    for word in unique_words:
        num_documents_containing_word = sum([1 for doc in documents if word in doc])
        idf[word] = math.log((1 + total_documents) / (1 + num_documents_containing_word)) + 1
    return idf

idf = compute_idf(filtered_documents, unique_words)

# Step 8: Compute TFIDF
tfidf_documents = []
for tf_doc in tf_documents:
    tfidf_doc = {word: tf_doc[word] * idf[word] for word in tf_doc}
    tfidf_documents.append(tfidf_doc)

# Step 9: Normalize TFIDF
def normalize_tfidf(tfidf_doc):
    norm = np.linalg.norm(list(tfidf_doc.values()))
    normalized_tfidf = {word: tfidf_doc[word] / norm for word in tfidf_doc}
    return normalized_tfidf

normalized_tfidf_documents = [normalize_tfidf(doc) for doc in tfidf_documents]

print(normalized_tfidf_documents)
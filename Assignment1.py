from bs4 import BeautifulSoup
import requests
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Fetch HTML content from URL
url = "https://en.wikipedia.org/wiki/The_Vampire_Diaries"
html_content = requests.get(url).text

# Extract text from specified HTML tags
soup = BeautifulSoup(html_content, 'html.parser')
text_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'a'])
text = ' '.join(tag.get_text() for tag in text_tags)

# Clean data
cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)

# Normalize text to lowercase
normalized_text = cleaned_text.lower()

# Tokenize text
tokens = word_tokenize(normalized_text)

# Lemmatize tokens
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in lemmatized_tokens if word not in stop_words]

# Extract unique words
unique_words = set(filtered_tokens)

# Print unique words
print(unique_words)
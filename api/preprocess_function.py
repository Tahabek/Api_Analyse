import re
import string
import json
import os
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(tweet: str, stem_or_lem: str = "lem", emojis=None, stopwords_list=None) -> str:
    """
    Text preprocessing:
    - Clean URLs, mentions, hashtags
    - Replace emojis with descriptions (if provided)
    - Lowercase + remove punctuation
    - Tokenization and stopword removal
    - Lemmatization or stemming
    - Remove non-ASCII characters (emojis etc.)
    """

    # 1. Basic cleaning
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)  # URLs
    tweet = re.sub(r"@\w+", '', tweet)  # Mentions
    tweet = re.sub(r"#", '', tweet)     # Hashtags

    # 2. Emoji replacement (if emojis dict provided)
    if emojis:
        for emo, desc in emojis.items():
            tweet = tweet.replace(emo, f" {desc} ")

    # 3. Remove non-ASCII characters (e.g., emojis not replaced)
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

    # 4. Lowercase and remove punctuation
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    # 5. Tokenization
    tokens = tweet.split()

    # 6. Stopwords removal (if list provided)
    if stopwords_list:
        tokens = [w for w in tokens if w not in stopwords_list]

    # 7. Lemmatization or stemming
    if stem_or_lem == "lem":
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    else:
        tokens = [stemmer.stem(w) for w in tokens]

    # 8. Final clean-up
    return ' '.join(tokens).strip()

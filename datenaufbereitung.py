import nltk
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
german_stop_words = set(stopwords.words('german'))

bereinigt = []

with open('lak_report.csv', 'r', encoding='utf-8') as csv:
    for zeile in csv:
        tokens = word_tokenize(zeile.lower())
        # Zahlen und Satzzeichen entfernen
        tokens_bereinigt = [token for token in tokens if token.isalpha()]
        # Stopwörter entfernen
        tokens = [wort for wort in tokens_bereinigt if wort not in german_stop_words]
        if len(tokens) > 3:
            bereinigt.extend(tokens)
            print(tokens)
            print(len(tokens))

#print(bereinigt)
worthaeufigkeit = nltk.FreqDist(bereinigt)
print(worthaeufigkeit.most_common(20))







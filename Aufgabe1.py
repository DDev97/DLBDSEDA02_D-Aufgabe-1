import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Download der Stopwörter
nltk.download('stopwords')
# Auswahl der deutschen Stopwörter
german_stop_words = set(stopwords.words('german'))

# Funktion für die Textvorverarbeitung
def text_vorverarbeitung(text):
    # Texte werden tokenisiert und in Kleinbuchstaben umgewandelt
    tokens = word_tokenize(text.lower())
    # Nur alphabetische Zeichen behalten und die Stopwärter entfernen
    tokens = [token for token in tokens if token.isalpha() and token not in german_stop_words]
    # Token werden zu einem Text zusammengefügt
    return ' '.join(tokens)

# Leere Liste erstellen
bereinigte_texte = []

# CSV Datei einlesen und Texte bereinigen
with open('lak_report.csv', 'r', encoding='utf-8') as csv:
    for zeile in csv:
        bereinigte_text = text_vorverarbeitung(zeile)
        bereinigte_texte.append(bereinigte_text)

# Count Vectorisierung
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(bereinigte_texte)

# Häufigste Wörter anzeigen
wort_haeufigkeit = np.sum(count_matrix.toarray(), axis=0)
wort_freq_pairs = list(zip(count_vectorizer.get_feature_names_out(), wort_haeufigkeit))
sorted_word_freq = sorted(wort_freq_pairs, key=lambda x: x[1], reverse=True)

# Ausgabe der Top 20 der am häufigsten Wörter
print("Top 20 der häufigsten Wörter:")
for wort, freq in sorted_word_freq[:20]:
    print(f"{wort}: {freq}")

# Vectorisierung mittels TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(bereinigte_texte)

# TF-IDF Scores für wichtige Begriffe erstellen und ausgeben
print("\nWichtige Begriffe nach TF-IDF:")
tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
tfidf_pairs = list(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_scores))
sorted_tfidf = sorted(tfidf_pairs, key=lambda x: x[1], reverse=True)
for wort, score in sorted_tfidf[:20]:
    print(f"{wort}: {score:.4f}")

# Zusammenfügen aller bereinigten Texte zu einem String,
# sonst können die Wörter nicht gelesen werden.
gesamter_text = ' '.join(bereinigte_texte)

# Entfernung zusätzlicher Wörter, um das WordCloud Bild zu optimieren
zusaetzliche_stopwoerter = {'wurde', 'original', 'name'}
german_stop_words.update(zusaetzliche_stopwoerter)

# Erstellung des WordCloud Bildes
wordcloud = WordCloud(
    width=1920,
    height=1080,
    background_color='white',
    max_words=50,
    collocations=True,
    stopwords=german_stop_words,
    random_state=42  # für reproduzierbare Ergebnisse
).generate(gesamter_text)

# Ausgabe des WordCloud Bildes
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Themen-Analyse mit WordCloud', fontsize=50)
plt.show()

#LDA
# Count Vectorisierung
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(bereinigte_texte)
# LDA Modell erstellen mit 10 Themen
lda = LatentDirichletAllocation(n_components=10, random_state=42)
# Modell trainieren
lda_model = lda.fit(dtm)

# 10 Themen ausgeben mit jeweils 10 Wörtern ausgeben
feature_names = vectorizer.get_feature_names_out()
print("\nGefundene Themen:")
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-10-1:-1]  # Top 10 Wörter
    top_words = [feature_names[i] for i in top_words_idx]
    print(f'Thema {topic_idx + 1}:')
    print(', '.join(top_words))

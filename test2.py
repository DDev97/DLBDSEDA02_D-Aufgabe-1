import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
review_1="Behandlung war gut, Ärzte und Pfleger waren freundlich"
review_2="Pfleger waren sehr freundlich die Ärzte waren kompetent"
vect = CountVectorizer()
data = vect.fit_transform([review_1,review_2])
data = pd.DataFrame(data.toarray(),columns=vect.get_feature_names_out())
print(data)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
review_1 = "Loved the sound, no battery issues test"
review_2 = "Sound quality is good; battery life not good test"
review_3 = "loved the battery, sound is bad"
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform([review_1, review_2, review_3])
data=pd.DataFrame(model.toarray(),columns=vectorizer.get_feature_names_out())
print(data)
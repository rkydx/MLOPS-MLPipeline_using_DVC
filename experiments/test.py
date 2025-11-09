from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love coding"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

print("Words:", vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())

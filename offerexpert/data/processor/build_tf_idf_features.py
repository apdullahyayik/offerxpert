from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get the feature names (words or terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array for easier inspection
tfidf_matrix_array = tfidf_matrix.toarray()

# Display the TF-IDF values for each term in each document
for i, document in enumerate(documents):
    print(f"Document {i + 1}: {document}")
    for j, term in enumerate(feature_names):
        tfidf_value = tfidf_matrix_array[i][j]
        if tfidf_value > 0:
            print(f"{term}: {tfidf_value}")
    print("\n")

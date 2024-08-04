import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example list of common questions
common_questions = [
    "Hi",
    "Hello",
    "Thank you",
    "How are you?",
    "What is your name?"
]

# Initialize a TF-IDF Vectorizer for similarity matching
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(common_questions)

def is_common_question(query):
    # Transform the query using the vectorizer
    query_vec = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarity between the query and the common questions
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Determine if the query is similar to any common question (using a threshold)
    return max(similarities) >= 0.9
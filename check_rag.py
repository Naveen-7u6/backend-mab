import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example list of common questions
common_questions = [
    "Hi",
    "Hello",
    "Thank you",
    "How are you?",
    "What is your name?"]

# Example keywords for RAG
rag_keywords = ["places", "tourist", "resorts", "stays", "holidays","vacations"]

# Keywords related to bookings
booking_keywords = ["flight", "book", "booking", "reserve", "reservation", "ticket", "schedule", "availability"]

# Initialize a TF-IDF Vectorizer for similarity matching
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(common_questions)

def needs_rag(query):
    # Keyword matching for RAG
    if any(keyword in query.lower() for keyword in rag_keywords):
        return True

    # # Similarity matching for common questions
    # query_vec = tfidf_vectorizer.transform([query])  # Use the vectorizer object to transform the query
    # similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # if max(similarities) < 0.5:  # Threshold for non-common questions
    #     return True

    # If no conditions are met, assume RAG is not needed
    return False

def needs_agent(query):
        # Check for booking-related keywords
    if any(keyword in query.lower() for keyword in booking_keywords):
        return True

    # Similarity matching for common questions
    # query_vec = tfidf_vectorizer.transform([query])  # Use the vectorizer object to transform the query
    # similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # if max(similarities) < 0.5:  # Threshold for non-common questions
    #     return True

    # If no conditions are met, assume RAG is not needed
    return False


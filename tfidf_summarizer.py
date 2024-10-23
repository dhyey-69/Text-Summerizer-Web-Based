import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download the necessary NLTK data
nltk.download('punkt')

def tfidf_summary(text, num_sentences=2):
    """
    Generate an extractive summary using TF-IDF.

    Parameters:
    - text (str): The input text to summarize.
    - num_sentences (int): Number of sentences to include in the summary.

    Returns:
    - str: The extractive summary.
    """
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Create a TF-IDF Vectorizer and fit_transform the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Sum the TF-IDF scores for each sentence
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Get the indices of the top N sentences
    top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
    
    # Sort the indices to maintain the original order of sentences
    top_sentence_indices.sort()
    
    # Construct the summary
    summary = ' '.join([sentences[i] for i in top_sentence_indices])
    return summary


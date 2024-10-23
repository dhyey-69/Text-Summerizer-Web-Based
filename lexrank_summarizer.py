import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def lexrank_summary(text, num_sentences=2):
    
    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Create a TF-IDF Vectorizer and fit_transform the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Compute the similarity matrix
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

    # Create a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # Compute PageRank scores for the sentences
    scores = nx.pagerank(nx_graph)

    # Get the indices of the top N sentences
    top_sentence_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]

    # Sort the indices to maintain the original order of sentences
    top_sentence_indices.sort()

    # Construct the summary
    summary = ' '.join([sentences[i] for i in top_sentence_indices])
    return summary


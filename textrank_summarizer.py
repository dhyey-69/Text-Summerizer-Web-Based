import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

# Download the necessary NLTK data
nltk.download('punkt')

def textrank_summary(text, num_sentences=2):
    
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)

    # Create a TF-IDF Vectorizer and fit_transform the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Compute cosine similarity matrix
    cosine_matrix = cosine_similarity(tfidf_matrix)

    # Build graph
    graph = nx.Graph()
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if cosine_matrix[i][j] > 0:  # Create edge if similarity is greater than 0
                graph.add_edge(i, j, weight=cosine_matrix[i][j])
    
    # Apply PageRank algorithm on the graph
    scores = nx.pagerank(graph)

    # Rank sentences by their score
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Select the top N sentences for the summary
    top_sentences = [ranked_sentences[i][1] for i in range(num_sentences)]
    
    # Return the summary (top sentences in their original order)
    return ' '.join(top_sentences)

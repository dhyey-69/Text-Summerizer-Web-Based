from flask import Flask, request, jsonify
from flask_cors import CORS
from tfidf_summarizer import tfidf_summary
from textrank_summarizer import textrank_summary 
from lexrank_summarizer import lexrank_summary
from t5_summarizer import t5_summary

app = Flask(__name__)
CORS(app)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    num_sentences = int(data.get('num_sentences', 2))
    technique = data.get('technique', 'tfidf')

    try:
        if technique == 'tfidf':
            summary = tfidf_summary(text, num_sentences)
        elif technique == 'textrank':
            summary = textrank_summary(text, num_sentences)
        elif technique == 'lexrank':
            summary = lexrank_summary(text, num_sentences)
        elif technique == 't5':
            summary = t5_summary(text, num_sentences)
        else:
            raise ValueError("Invalid summarization technique selected.")
        
        return jsonify({'summary': summary})
    except Exception as e:
        print(f"Error occurred during summarization: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

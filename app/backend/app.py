"""
Flask API for sentiment analysis
Provides REST endpoints for the sentiment analyzer
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentiment_analyzer import SentimentAnalyzer
import os

app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Enable CORS for frontend

# Initialize analyzer
analyzer = SentimentAnalyzer()


@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of input text"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Analyze text
        word_results = analyzer.analyze_text(text)
        overall = analyzer.get_overall_sentiment(word_results)

        return jsonify({
            'success': True,
            'text': text,
            'words': word_results,
            'overall': overall
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example sentences for demo"""
    examples = [
        {
            "text": "I love this amazing movie it was fantastic",
            "category": "Positive"
        },
        {
            "text": "This is terrible and awful I hate it",
            "category": "Negative"
        },
        {
            "text": "The food was okay but the service was great",
            "category": "Mixed"
        },
        {
            "text": "Best product ever absolutely wonderful experience",
            "category": "Positive"
        },
        {
            "text": "Worst experience horrible and disappointing",
            "category": "Negative"
        },
        {
            "text": "It was fine nothing special but not bad",
            "category": "Neutral"
        }
    ]

    return jsonify({
        'success': True,
        'examples': examples
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'model': 'Deep MEMM Sentiment Analyzer'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

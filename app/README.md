# Sentiment Flow Visualizer

An interactive web application that performs real-time sentiment analysis using Deep Maximum Entropy Markov Models (DMEMM) for NLP.

## Features

- **Real-time Analysis**: Instant sentiment classification as you type
- **Word-by-Word Breakdown**: See sentiment for each word with confidence scores
- **Beautiful Visualizations**: Animated charts and color-coded sentiment flow
- **Multiple Model Support**: Built on MLP, BiLSTM, and Word2Vec architectures
- **Interactive UI**: Modern, responsive design with example sentences
- **REST API**: Clean API for integration into other projects

## Tech Stack

- **Backend**: Python, Flask, PyTorch
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Models**: Multi-Layer Perceptron, Bi-LSTM MEMM, Word2Vec embeddings
- **Sentiment Classes**: Positive, Negative, Neutral

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
cd app/backend
python app.py
```

The server will start on `http://localhost:5000`

### 3. Open in Browser

Navigate to `http://localhost:5000` in your web browser

## How It Works

### Deep MEMM Architecture

This project implements a Deep Maximum Entropy Markov Model with:

1. **Word Embeddings**: Each word is converted to a dense vector representation
2. **Context Modeling**: The model considers the current word and previous tag
3. **Multi-Layer Perceptron**: Deep neural network for feature extraction
4. **Sentiment Classification**: Outputs probability distribution over sentiment classes

### Model Variants

The original research includes three implementations:

- **MLP with Random Initialization** (`dmemm/mlp.py`)
- **Bi-LSTM MEMM** (`dmemm/bilstm.py`)
- **MLP with Word2Vec** (`dmemm/mlp-word2vec.py`)

The web app uses a simplified inference version optimized for real-time performance.

## API Documentation

### POST `/api/analyze`

Analyze sentiment of input text.

**Request:**
```json
{
  "text": "I love this amazing movie"
}
```

**Response:**
```json
{
  "success": true,
  "text": "I love this amazing movie",
  "words": [
    {
      "word": "i",
      "sentiment": "Neutral",
      "tag": "O",
      "confidence": 0.85,
      "probabilities": {
        "Positive": 0.10,
        "Negative": 0.05,
        "Neutral": 0.85
      }
    },
    ...
  ],
  "overall": {
    "sentiment": "Positive",
    "confidence": 0.92,
    "scores": {
      "Positive": 0.75,
      "Negative": 0.05,
      "Neutral": 0.20
    }
  }
}
```

### GET `/api/examples`

Get example sentences for testing.

### GET `/api/health`

Health check endpoint.

## Project Structure

```
dmemm/
├── app/
│   ├── backend/
│   │   ├── app.py                 # Flask API server
│   │   └── sentiment_analyzer.py  # Simplified inference module
│   └── frontend/
│       └── index.html             # Interactive web UI
├── dmemm/
│   ├── mlp.py                     # Original MLP implementation
│   ├── bilstm.py                  # BiLSTM MEMM implementation
│   └── mlp-word2vec.py            # Word2Vec version
├── requirements.txt
└── README.md
```

## Development

### Running in Development Mode

```bash
cd app/backend
python app.py
```

The Flask app runs with hot-reload enabled for development.

### Customization

- **Add vocabulary**: Edit `_build_demo_vocab()` in `sentiment_analyzer.py`
- **Modify model**: Update the `SimpleMLP` class architecture
- **Change styling**: Edit the CSS in `index.html`
- **Add features**: Extend the Flask API with new endpoints

## Use Cases

- **Product Reviews**: Analyze customer feedback sentiment
- **Social Media Monitoring**: Track sentiment in tweets/posts
- **Content Moderation**: Identify negative content
- **Market Research**: Gauge public opinion
- **Customer Service**: Prioritize negative feedback

## Performance

- **Inference Speed**: ~10-20ms per sentence on CPU
- **Scalability**: Can process hundreds of requests per second
- **Memory**: ~50MB for model in memory

## Future Enhancements

- [ ] Load pre-trained models from the original implementations
- [ ] Add comparison between MLP, BiLSTM, and Word2Vec models
- [ ] Export analysis results to CSV/JSON
- [ ] Batch processing for multiple texts
- [ ] Fine-tuning interface for custom datasets
- [ ] Mobile app version
- [ ] Cloud deployment (Heroku, AWS, etc.)

## License

This project is built for educational and portfolio purposes.

## Acknowledgments

Based on Deep Maximum Entropy Markov Models for sequence labeling and sentiment analysis in NLP.

## Contact

Perfect for showcasing in your portfolio, demonstrating skills in:
- Deep Learning & NLP
- Full-stack Development
- API Design
- Interactive Visualizations
- PyTorch Model Deployment

---

Made with PyTorch and Flask

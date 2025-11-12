# Sentiment Flow Visualizer

**Deep Maximum Entropy Markov Model for NLP**

An interactive web application showcasing real-time sentiment analysis using Deep Learning and NLP.

![Portfolio Project](https://img.shields.io/badge/Portfolio-Project-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-red)
![Flask](https://img.shields.io/badge/Flask-API-green)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)

## Live Demo

**Try it now**: Run `./run_demo.sh` or `python demo.py` for a quick demo!

## Features

- **Interactive Web Interface**: Beautiful, modern UI with real-time sentiment analysis
- **Word-by-Word Analysis**: See sentiment classification for each word with confidence scores
- **Visual Analytics**: Animated charts and color-coded sentiment flow
- **REST API**: Clean API for integration into other projects
- **Multiple Neural Architectures**: MLP, BiLSTM, Word2Vec embeddings

## Quick Start

### Option 1: Web App (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
./run_demo.sh
```

Then open `http://localhost:5000` in your browser!

### Option 2: Command Line Demo

```bash
python demo.py
```

## Project Structure

```
dmemm/
├── app/                        # Portfolio Web Application
│   ├── backend/
│   │   ├── app.py             # Flask REST API
│   │   └── sentiment_analyzer.py  # Inference module
│   └── frontend/
│       └── index.html         # Interactive UI
├── dmemm/                      # Original Research Implementations
│   ├── mlp.py                 # Multi-Layer Perceptron
│   ├── bilstm.py              # Bi-LSTM MEMM
│   └── mlp-word2vec.py        # Word2Vec embeddings
├── demo.py                     # Quick CLI demo
├── run_demo.sh                 # One-click launcher
└── requirements.txt

```

## Technical Details

### Deep MEMM Architecture

This project implements sentiment analysis using:

1. **Maximum Entropy Markov Models (MEMM)**: Conditional probabilistic sequence model
2. **Neural Network Features**: Deep learning for feature extraction
3. **Context Modeling**: Considers word context and previous predictions
4. **Sentiment Classes**: Positive (T-POS), Negative (T-NEG), Neutral (T-NEU, O)

### Model Implementations

- **MLP with Random Init** (`dmemm/mlp.py`): 15-dim embeddings, 128-hidden units
- **Bi-LSTM MEMM** (`dmemm/bilstm.py`): Bidirectional LSTM with Viterbi decoding
- **MLP with Word2Vec** (`dmemm/mlp-word2vec.py`): 300-dim pre-trained embeddings

### API Usage

```python
# POST /api/analyze
{
  "text": "I love this amazing movie!"
}

# Response
{
  "success": true,
  "overall": {
    "sentiment": "Positive",
    "confidence": 0.92
  },
  "words": [...]
}
```

## Screenshots

### Main Interface
- Real-time text analysis
- Color-coded word tags
- Sentiment probability bars
- Overall sentiment with confidence

### Word-Level Analysis
- Individual word sentiments
- Confidence scores per word
- Emoji indicators
- Animated results

## Use Cases

- Product review sentiment analysis
- Social media monitoring
- Customer feedback analysis
- Content moderation
- Market research

## Technologies

- **Backend**: Python 3.7+, Flask, PyTorch
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **ML**: Neural Networks, Word Embeddings, Sequence Modeling
- **NLP**: Tokenization, Sentiment Classification, MEMM

## Performance

- **Inference**: ~10-20ms per sentence
- **Throughput**: Hundreds of requests/second
- **Memory**: ~50MB model footprint

## Portfolio Highlights

This project demonstrates:
- Deep Learning & NLP expertise
- Full-stack development (Flask + Frontend)
- REST API design
- Interactive data visualization
- Model deployment and inference optimization
- Clean, documented code

## Future Enhancements

- [ ] Model comparison interface (MLP vs BiLSTM vs Word2Vec)
- [ ] Fine-tuning on custom datasets
- [ ] Batch processing for multiple texts
- [ ] Export results to CSV/JSON
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, Heroku)
- [ ] Mobile-responsive improvements

## Development

See [app/README.md](app/README.md) for detailed development documentation.

## License

Educational and portfolio project.

## Acknowledgments

Based on Deep Maximum Entropy Markov Models for sequence labeling in NLP.

---

**Built with PyTorch, Flask, and passion for NLP**

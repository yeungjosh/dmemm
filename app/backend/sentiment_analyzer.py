"""
Simplified sentiment analyzer for inference
Converts the training scripts into a clean inference API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleMLP(nn.Module):
    """Simplified MLP for sentiment tagging"""

    def __init__(self, vocab_size, embedding_dim=15, context_size=3):
        super(SimpleMLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.sentiment_tags = ["T-POS", "T-NEG", "T-NEU", "O"]

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)

        # Network layers
        self.fc1 = nn.Linear((context_size - 1) * embedding_dim + 5, 128)
        self.act1 = nn.ReLU()
        self.out = nn.Linear(128, len(self.sentiment_tags))

    def forward(self, input_words, input_tags):
        if list(input_words.size()) == [1]:
            zeros = torch.zeros([1, self.embedding_dim])
            catted = torch.cat((zeros, self.embeddings(input_words).view((1, -1))), dim=1)
            embeds = catted.view((1, -1))
        else:
            embeds = self.embeddings(input_words).view((1, -1))

        combined = torch.cat((embeds.view(1, -1), input_tags.float().view(1, -1)), dim=1)
        a1 = self.fc1(combined)
        h1 = self.act1(a1)
        a2 = self.out(h1)
        log_probs = F.log_softmax(a2, dim=1)
        return log_probs


class SentimentAnalyzer:
    """Main sentiment analyzer class for easy inference"""

    def __init__(self):
        self.sentiment_tags = ["T-POS", "T-NEG", "T-NEU", "O"]
        self.tag_names = {
            "T-POS": "Positive",
            "T-NEG": "Negative",
            "T-NEU": "Neutral",
            "O": "Neutral"
        }
        self.tag_to_ix = {
            "T-POS": 0,
            "T-NEG": 1,
            "T-NEU": 2,
            "O": 3,
            "<START>": 4,
            "<STOP>": 5
        }
        self.one_hot_label = {
            'T-POS': [1, 0, 0, 0, 0],
            'T-NEG': [0, 1, 0, 0, 0],
            'T-NEU': [0, 0, 1, 0, 0],
            'O': [0, 0, 0, 1, 0],
            '<START>': [0, 0, 0, 0, 1]
        }

        # Simple vocabulary for demo
        self.word_to_ix = {}
        self._build_demo_vocab()

        # Unknown token
        self.UNK_TOKEN = "<UNK>"
        self.word_to_ix[self.UNK_TOKEN] = len(self.word_to_ix)
        self.unk_idx = self.word_to_ix[self.UNK_TOKEN]

        # Initialize model
        self.model = SimpleMLP(len(self.word_to_ix))
        self.model.eval()

    def _build_demo_vocab(self):
        """Build a demo vocabulary with common sentiment words"""
        demo_words = [
            # Positive words
            "love", "great", "awesome", "excellent", "amazing", "wonderful",
            "fantastic", "good", "best", "happy", "beautiful", "perfect",
            "brilliant", "outstanding", "superb", "delightful", "enjoyable",

            # Negative words
            "hate", "terrible", "awful", "bad", "worst", "horrible",
            "disgusting", "poor", "disappointing", "sad", "angry", "pathetic",
            "useless", "annoying", "frustrating", "boring", "ugly",

            # Neutral words
            "the", "a", "an", "is", "was", "are", "were", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "can", "may", "might", "and", "or", "but", "if", "then",
            "this", "that", "these", "those", "it", "they", "we", "i",
            "you", "he", "she", "my", "your", "his", "her", "our",

            # Common nouns/verbs
            "movie", "film", "food", "service", "product", "place", "time",
            "day", "night", "work", "make", "see", "get", "go", "come",
            "think", "know", "want", "need", "like", "look", "use", "find"
        ]

        for word in demo_words:
            if word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)

    def analyze_text(self, text):
        """
        Analyze sentiment of text word by word

        Args:
            text: Input text string

        Returns:
            List of dicts with word and sentiment info
        """
        words = text.lower().split()
        results = []

        prev_tag = "<START>"

        with torch.no_grad():
            for i, word in enumerate(words):
                # Get word index (or use unknown token)
                word_idx = self.word_to_ix.get(word, self.unk_idx)

                # Prepare context
                if i == 0:
                    # First word: only current word
                    context_word_idxs = torch.tensor([word_idx], dtype=torch.long)
                else:
                    # Subsequent words: previous + current
                    prev_word = words[i-1]
                    prev_word_idx = self.word_to_ix.get(prev_word, self.unk_idx)
                    context_word_idxs = torch.tensor([prev_word_idx, word_idx], dtype=torch.long)

                # Previous tag one-hot
                context_tag_idxs = torch.tensor(self.one_hot_label[prev_tag], dtype=torch.long)

                # Get prediction
                log_probs = self.model(context_word_idxs, context_tag_idxs)
                probs = torch.exp(log_probs).squeeze().tolist()

                # Get predicted tag
                pred_idx = torch.argmax(log_probs, dim=1).item()
                pred_tag = self.sentiment_tags[pred_idx]

                # Store result
                results.append({
                    'word': word,
                    'sentiment': self.tag_names.get(pred_tag, 'Neutral'),
                    'tag': pred_tag,
                    'confidence': probs[pred_idx],
                    'probabilities': {
                        'Positive': probs[0],
                        'Negative': probs[1],
                        'Neutral': max(probs[2], probs[3])
                    }
                })

                prev_tag = pred_tag

        return results

    def get_overall_sentiment(self, results):
        """Calculate overall sentiment from word-level results"""
        if not results:
            return {'sentiment': 'Neutral', 'confidence': 0.0}

        # Aggregate sentiments (weighted by confidence)
        total_pos = sum(r['probabilities']['Positive'] for r in results)
        total_neg = sum(r['probabilities']['Negative'] for r in results)
        total_neu = sum(r['probabilities']['Neutral'] for r in results)

        total = total_pos + total_neg + total_neu
        if total == 0:
            return {'sentiment': 'Neutral', 'confidence': 0.0}

        # Normalize
        scores = {
            'Positive': total_pos / total,
            'Negative': total_neg / total,
            'Neutral': total_neu / total
        }

        # Get dominant sentiment
        dominant = max(scores.items(), key=lambda x: x[1])

        return {
            'sentiment': dominant[0],
            'confidence': dominant[1],
            'scores': scores
        }


# Demo function
def demo():
    analyzer = SentimentAnalyzer()

    test_sentences = [
        "I love this amazing movie",
        "This is terrible and awful",
        "The food was okay but the service was great",
        "I hate waiting in long lines"
    ]

    for sentence in test_sentences:
        print(f"\n📝 Text: {sentence}")
        results = analyzer.analyze_text(sentence)

        print("Word-by-word analysis:")
        for r in results:
            emoji = "😊" if r['sentiment'] == 'Positive' else "😞" if r['sentiment'] == 'Negative' else "😐"
            print(f"  {emoji} '{r['word']}': {r['sentiment']} ({r['confidence']:.2f})")

        overall = analyzer.get_overall_sentiment(results)
        print(f"\n🎯 Overall: {overall['sentiment']} ({overall['confidence']:.2f})")


if __name__ == "__main__":
    demo()

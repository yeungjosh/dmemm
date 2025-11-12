#!/usr/bin/env python3
"""
Quick demo of the sentiment analyzer
Run this to test the backend without starting the web server
"""

import sys
sys.path.append('app/backend')

from sentiment_analyzer import SentimentAnalyzer

def print_divider():
    print("\n" + "="*70 + "\n")

def demo():
    print_divider()
    print("   SENTIMENT FLOW VISUALIZER - DEMO")
    print("   Deep Maximum Entropy Markov Model for NLP")
    print_divider()

    # Initialize analyzer
    print("Loading sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    print("Model loaded successfully!\n")

    # Test sentences
    test_sentences = [
        "I love this amazing movie it was fantastic",
        "This is terrible and awful I hate it",
        "The food was okay but the service was great",
        "Best product ever absolutely wonderful experience",
        "Worst experience horrible and disappointing",
        "It was fine nothing special but not bad"
    ]

    for sentence in test_sentences:
        print_divider()
        print(f"📝 Text: {sentence}")
        print("-" * 70)

        # Analyze
        results = analyzer.analyze_text(sentence)

        # Word-by-word analysis
        print("\nWord-by-word analysis:")
        for r in results:
            emoji = "😊" if r['sentiment'] == 'Positive' else "😞" if r['sentiment'] == 'Negative' else "😐"
            confidence_bar = "█" * int(r['confidence'] * 20)
            print(f"  {emoji} '{r['word']:12s}' → {r['sentiment']:8s} [{confidence_bar:20s}] {r['confidence']:.2f}")

        # Overall sentiment
        overall = analyzer.get_overall_sentiment(results)
        print(f"\n🎯 Overall Sentiment: {overall['sentiment']}")
        print(f"   Confidence: {overall['confidence']:.2f}")

        if 'scores' in overall:
            print(f"\n   Score Breakdown:")
            print(f"   • Positive: {overall['scores']['Positive']:.2f}")
            print(f"   • Negative: {overall['scores']['Negative']:.2f}")
            print(f"   • Neutral:  {overall['scores']['Neutral']:.2f}")

    print_divider()
    print("✅ Demo complete!")
    print("\nTo run the web app:")
    print("  1. cd app/backend")
    print("  2. python app.py")
    print("  3. Open http://localhost:5000 in your browser")
    print_divider()


if __name__ == "__main__":
    demo()

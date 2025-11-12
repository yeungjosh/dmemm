# DMEMM - Deep Maximum Entropy Markov Model for NLP

A comprehensive implementation of Maximum Entropy Markov Models (MEMM) using deep learning approaches for sentiment analysis and sequence tagging tasks. This project explores three different neural architectures for capturing contextual information in text sequences.

## Table of Contents
- [Overview](#overview)
- [Background: Maximum Entropy Markov Models](#background-maximum-entropy-markov-models)
- [Model Architectures](#model-architectures)
  - [Option 1: MLP with Random Embeddings](#option-1-mlp-with-random-embeddings)
  - [Option 2: MLP with Word2Vec Embeddings](#option-2-mlp-with-word2vec-embeddings)
  - [Option 3: BiLSTM-MEMM](#option-3-bilstm-memm)
- [Data Flow & Preprocessing](#data-flow--preprocessing)
- [Training Process](#training-process)
- [Results & Model Comparison](#results--model-comparison)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Key Insights](#key-insights)

---

## Overview

This project implements three variants of deep learning models for sequence labeling, specifically targeting sentiment analysis with the following tags:
- **T-POS**: Positive sentiment
- **T-NEG**: Negative sentiment
- **T-NEU**: Neutral sentiment
- **O**: No sentiment (other)

The models combine traditional MEMM approaches with modern deep learning techniques to capture sequential dependencies and context in text data.

---

## Background: Maximum Entropy Markov Models

**MEMMs** are discriminative sequence models that predict each label conditioned on:
1. **Observations** (words/features)
2. **Previous state** (previous tag)

Unlike HMMs which model joint probability P(words, tags), MEMMs directly model conditional probability:

```
P(tag_i | word_i, tag_{i-1}, context)
```

This allows MEMMs to incorporate rich, overlapping features and avoid independence assumptions.

### MEMM Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMM Sequence Tagging                     │
└─────────────────────────────────────────────────────────────┘

Input Sentence: ["love", "this", "movie", "but", "hate", "ending"]

Step 1:          Step 2:          Step 3:
┌─────────┐     ┌─────────┐     ┌─────────┐
│  "love" │     │  "this" │     │ "movie" │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     │   ┌─────┐     │   ┌─────┐     │   ┌─────┐
     └──→│START│     └──→│T-POS│     └──→│T-POS│
         └──┬──┘         └──┬──┘         └──┬──┘
            │               │               │
            ▼               ▼               ▼
        ┌───────┐       ┌───────┐       ┌───────┐
        │ T-POS │       │ T-POS │       │ T-NEU │
        └───────┘       └───────┘       └───────┘
      (Predicted)     (Predicted)     (Predicted)

Each prediction uses:
  • Current word embedding
  • Previous predicted tag
  • Context words (n-gram or LSTM)
```

---

## Model Architectures

### Option 1: MLP with Random Embeddings

**File**: `dmemm/mlp.py`

This approach learns word embeddings from scratch during training.

#### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│           MLP with Random Initialized Embeddings                  │
└──────────────────────────────────────────────────────────────────┘

Input: Bigram Context with Previous Tag
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  word_{i-1}     word_i       tag_{i-1}
     │              │              │
     │              │              │
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌──────────┐
│Embedding│   │Embedding│   │ One-Hot  │
│  Layer  │   │  Layer  │   │ Encoding │
│ (15-dim)│   │ (15-dim)│   │  (5-dim) │
└────┬────┘   └────┬────┘   └─────┬────┘
     │              │              │
     └──────┬───────┘              │
            │                      │
            ▼                      │
     ┌────────────┐                │
     │ Concatenate│◄───────────────┘
     │  (30 + 5)  │
     └─────┬──────┘
           │ 35-dimensional vector
           ▼
     ┌───────────┐
     │  Linear   │
     │ (35→128)  │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │   ReLU    │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │  Linear   │
     │  (128→4)  │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │ LogSoftmax│
     └─────┬─────┘
           │
           ▼
    [T-POS, T-NEG, T-NEU, O]
    (Tag probabilities)
```

#### Key Features
- **Embedding Dimension**: 15
- **Context Size**: Bigram (previous word + current word)
- **Hidden Layer**: 128 units
- **Learns embeddings**: Embeddings are randomly initialized and trained end-to-end

#### When to Use
- Domain-specific vocabulary not in pre-trained models
- Twitter text, medical terminology, or specialized jargon
- When you have sufficient training data to learn good embeddings

---

### Option 2: MLP with Word2Vec Embeddings

**File**: `dmemm/mlp-word2vec.py`

This approach uses pre-trained Google News Word2Vec embeddings (300-dimensional).

#### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│              MLP with Pre-trained Word2Vec                        │
└──────────────────────────────────────────────────────────────────┘

Pre-trained Word2Vec Model (GoogleNews-vectors-negative300.bin)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ↓ (Frozen weights)

  word_{i-1}     word_i       tag_{i-1}
     │              │              │
     │              │              │
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌──────────┐
│ Word2Vec│   │ Word2Vec│   │ One-Hot  │
│  Lookup │   │  Lookup │   │ Encoding │
│(300-dim)│   │(300-dim)│   │  (5-dim) │
└────┬────┘   └────┬────┘   └─────┬────┘
     │              │              │
     │  If word not in vocab:      │
     │  use zero vector            │
     │              │              │
     └──────┬───────┘              │
            │                      │
            ▼                      │
     ┌────────────┐                │
     │ Concatenate│◄───────────────┘
     │(600 + 5)   │
     └─────┬──────┘
           │ 605-dimensional vector
           ▼
     ┌───────────┐
     │  Linear   │
     │ (605→300) │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │   ReLU    │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │  Linear   │
     │ (300→300) │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │   ReLU    │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │  Linear   │
     │  (300→4)  │
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │ LogSoftmax│
     └─────┬─────┘
           │
           ▼
    [T-POS, T-NEG, T-NEU, O]
    (Tag probabilities)
```

#### Pre-trained Embeddings Handling

```
Word Vocabulary Handling
━━━━━━━━━━━━━━━━━━━━━━━

Input Word
    │
    ▼
┌─────────────────┐
│ Word in W2V?    │
└────┬───────┬────┘
     │Yes    │No
     │       │
     ▼       ▼
 ┌─────┐  ┌──────────┐
 │ W2V │  │ Zero Vec │
 │ Vec │  │ (300-dim)│
 └─────┘  └──────────┘
     │         │
     └────┬────┘
          │
          ▼
    300-dim Vector
```

#### Key Features
- **Embedding Dimension**: 300 (pre-trained)
- **Word2Vec Model**: GoogleNews-vectors-negative300 (first 50,000 words)
- **Frozen Embeddings**: Pre-trained vectors are not updated during training
- **Out-of-vocabulary**: Words not in Word2Vec get zero vectors
- **Network Depth**: 3 fully connected layers (300 → 300 → 4)

#### When to Use
- **Small datasets**: Leverage knowledge from large corpora
- **General vocabulary**: Standard English words
- **Best performance**: According to results, this option performed best

---

### Option 3: BiLSTM-MEMM

**File**: `dmemm/bilstm.py`

This approach uses a Bidirectional LSTM to capture context from the entire sentence.

#### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    BiLSTM-MEMM Architecture                       │
└──────────────────────────────────────────────────────────────────┘

Input Sentence: [word_1, word_2, ..., word_n]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Sentence Encoding
─────────────────────────

word_1    word_2    word_3    ...    word_n
   │         │         │              │
   ▼         ▼         ▼              ▼
┌──────┐ ┌──────┐ ┌──────┐        ┌──────┐
│Embed │ │Embed │ │Embed │        │Embed │
│15-dim│ │15-dim│ │15-dim│   ...  │15-dim│
└──┬───┘ └──┬───┘ └──┬───┘        └──┬───┘
   │         │         │              │
   └────┬────┴────┬────┴──────────────┘
        │         │
        ▼         ▼
    ┌─────────────────────────────────┐
    │   Bidirectional LSTM Layer      │
    │                                 │
    │  Forward →  →  →  →  →  →  →  │
    │                                 │
    │  ← ← ← ← ← ← ← Backward        │
    └────┬───┬───┬───────────┬────────┘
         │   │   │           │
         ▼   ▼   ▼           ▼
      ┌────┐┌────┐       ┌────┐
      │ h1 ││ h2 │  ...  │ hn │  (hidden states, 10-dim)
      └─┬──┘└─┬──┘       └─┬──┘
        │     │            │
        ▼     ▼            ▼
    ┌──────┐┌──────┐  ┌──────┐
    │Linear││Linear│  │Linear│
    │10→6  ││10→6  │  │10→6  │
    └──┬───┘└──┬───┘  └──┬───┘
       │      │         │
       ▼      ▼         ▼
    [feat1][feat2]...[featn]  (features for each word)


Step 2: MEMM Scoring with Viterbi Decoding
──────────────────────────────────────────

For each position i, compute transition scores:

    P(tag_i | features_i, tag_{i-1})


        tag_{i-1}       features_i
            │               │
            ▼               ▼
        ┌─────────────────────┐
        │  Transition Matrix  │
        │  + Feature Score    │
        └──────────┬──────────┘
                   │
                   ▼
            Score(tag_i)


Viterbi Algorithm:
─────────────────

Time:    t=0      t=1          t=2          t=3
       ┌─────┐  ┌─────┐      ┌─────┐      ┌─────┐
Tags:  │START│  │     │      │     │      │STOP │
       └──┬──┘  └──┬──┘      └──┬──┘      └─────┘
          │        │            │
          │    ┌───┼───┐    ┌───┼───┐
          │    │   │   │    │   │   │
          ▼    ▼   ▼   ▼    ▼   ▼   ▼
       ┌────┐┌────┐┌────┐┌────┐┌────┐
       │T-POS│T-NEG│T-NEU│  O  │T-POS│ ...
       └─┬──┘└─┬──┘└─┬──┘└─┬──┘└─┬──┘
         │     │     │     │     │
    Score│ Score │ Score │ Score │ ...
         │     │     │     │     │
         └─────┴─────┴─────┴─────┘
                   │
                   ▼
          Backtrack for best path
                   │
                   ▼
       [T-POS, T-POS, T-NEU, O, T-NEG]
              (Final prediction)
```

#### BiLSTM Detailed View

```
Bidirectional LSTM Cell Processing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At each timestep t:

Forward Direction (→):
─────────────────────
    h_{t-1}  x_t
       │      │
       └──┬───┘
          │
    ┌─────▼─────┐
    │ LSTM Cell │
    │  (forget, │
    │   input,  │
    │   output  │
    │   gates)  │
    └─────┬─────┘
          │
          ▼
        h_t →


Backward Direction (←):
──────────────────────
    h_{t+1}  x_t
       │      │
       └──┬───┘
          │
    ┌─────▼─────┐
    │ LSTM Cell │
    │  (forget, │
    │   input,  │
    │   output  │
    │   gates)  │
    └─────┬─────┘
          │
          ▼
        ← h_t


Combined:
────────
    h_t → ⊕ ← h_t
         │
         ▼
    [Concatenated
     bidirectional
     hidden state]
         │
         ▼
    Feature vector
    for position t
```

#### Key Features
- **Embedding Dimension**: 15 (randomly initialized)
- **Hidden Dimension**: 10 (5 per direction)
- **Bidirectional**: Captures context from both left and right
- **Viterbi Decoding**: Finds optimal tag sequence using dynamic programming
- **Transition Matrix**: Learned conditional probabilities P(tag_i | tag_{i-1})

#### MEMM Scoring Function

```python
# For each word position, compute:
score = feature_score(word_i) + transition_score(tag_{i-1} → tag_i)

# The model learns:
# 1. Feature scores from BiLSTM
# 2. Transition probabilities between tags
```

#### When to Use
- **Long-range dependencies**: Captures context from entire sentence
- **Better than n-grams**: Not limited to fixed window size
- **Structured prediction**: Viterbi ensures globally consistent tag sequences

---

## Data Flow & Preprocessing

### Data Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    Data Processing Pipeline                       │
└──────────────────────────────────────────────────────────────────┘

1. Load Raw Data
━━━━━━━━━━━━━━━
    train_set.pkl / test_set.pkl
            │
            ▼
    ┌────────────────┐
    │ List of dicts: │
    │ {              │
    │  'words': [...],│
    │  'ts_raw_tags':│
    │         [....]  │
    │ }              │
    └───────┬────────┘
            │
            ▼

2. Create Word-Tag Tuples
━━━━━━━━━━━━━━━━━━━━━━━━
    ([words], [tags])
            │
            ▼
    Example:
    (['love', 'this', 'movie'], ['T-POS', 'T-POS', 'O'])
            │
            ▼

3. Build N-gram Contexts (Options 1 & 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Bigram with Previous Tag:
    ([word_{i-1}, word_i, tag_{i-1}], tag_i)
            │
            ▼
    Example:
    (['love', 'this', 'T-POS'], 'T-POS')
    (['this', 'movie', 'T-POS'], 'O')
            │
            ▼

4. Flatten & Split
━━━━━━━━━━━━━━━━━
    All bigrams from all sentences
            │
            ├──→ 80% Train
            └──→ 20% Validation
            │
            ▼

5. Convert to Tensors
━━━━━━━━━━━━━━━━━━━━
    Options 1:
    word → index → embedding

    Option 2:
    word → Word2Vec vector (300-dim)

    Option 3:
    sentence → indices → embeddings → BiLSTM
            │
            ▼

6. Training
━━━━━━━━━━━
    Batch processing → Forward pass → Loss → Backprop
```

### Tag Encoding

```
Tag Encoding Scheme
━━━━━━━━━━━━━━━━━━━

Tag Name  │  One-Hot Encoding    │  Index
──────────┼──────────────────────┼────────
T-POS     │  [1, 0, 0, 0, 0]    │   0
T-NEG     │  [0, 1, 0, 0, 0]    │   1
T-NEU     │  [0, 0, 1, 0, 0]    │   2
O         │  [0, 0, 0, 1, 0]    │   3
<START>   │  [0, 0, 0, 0, 1]    │   4
<STOP>    │  N/A                 │   5
```

### Data Splits

```
Dataset Splits (80-20 split)
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original Sentences
        │
        ▼
┌───────────────────┐
│  Generate n-grams │
│  (or full sent.)  │
└────────┬──────────┘
         │
         ▼
┌────────────────────┐
│ Shuffle & Split    │
└─────┬──────────┬───┘
      │          │
      ▼          ▼
   Train      Validation
   (80%)        (20%)
```

---

## Training Process

### Training Loop

```
┌──────────────────────────────────────────────────────────────────┐
│                    Training Loop (Per Epoch)                      │
└──────────────────────────────────────────────────────────────────┘

FOR each epoch:
    │
    ├─→ FOR each training sample:
    │       │
    │       ├─→ 1. Prepare Input
    │       │       ├─ Convert words to embeddings
    │       │       ├─ Encode previous tag
    │       │       └─ Create input tensor
    │       │
    │       ├─→ 2. Forward Pass
    │       │       ├─ Pass through network
    │       │       └─ Get log probabilities
    │       │
    │       ├─→ 3. Compute Loss
    │       │       └─ NLL Loss between prediction and true tag
    │       │
    │       ├─→ 4. Backward Pass
    │       │       ├─ Compute gradients
    │       │       └─ Update parameters
    │       │
    │       └─→ 5. Track Loss
    │
    └─→ Return average epoch loss
```

### Loss Function

All three options use **Negative Log-Likelihood (NLL) Loss**:

```
NLL Loss
━━━━━━━━

Given:
- Predicted log probabilities: [log P(T-POS), log P(T-NEG), log P(T-NEU), log P(O)]
- True tag: T-POS (index 0)

Loss = -log P(T-POS)
     = -predicted_log_probs[0]

Goal: Minimize this loss
     → Maximize probability of correct tag
```

### Optimizers

```
Optimizer Configurations
━━━━━━━━━━━━━━━━━━━━━━

Option 1 & 2:
┌────────────────────┐
│  Adam Optimizer    │
│  lr = 0.001        │
│  β₁ = 0.9          │
│  β₂ = 0.999        │
└────────────────────┘

Option 3:
┌────────────────────┐
│  Adam Optimizer    │
│  lr = 0.01         │
│  (higher for LSTM) │
└────────────────────┘
```

### Learning Rate Impact (from Report)

```
Learning Rate Comparison (Option 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

lr = 0.001 (BEST):          lr = 0.01:              lr = 0.05:
    Loss                         Loss                     Loss
     │                            │                         │
  90 ┤●                        300┤  ●                  20000┤●
     │ ●                          │ ● ●                      │
  80 ┤  ●                      250┤    ●                     │
     │   ●                        │  ●   ●              15000┤
  70 ┤    ●                    200┤       ●                  │
     │     ●                      │    ●    ●                │
  60 ┤      ●●                 150┤         ●           10000┤
     │        ●                   │ ●  ●      ●              │
  50 ┤         ●●              100┤           ●              │
     │           ●                │      ●      ●        5000┤
  40 ┤            ●●            50┤               ●●         │
     │              ●              │                   ●      │
  30 ┤               ●●●●●●●●    0┼─────────────────●●●●● 0┼──────────────────
     └────────────────────        └─────────────────         └─────────────────
     0    10   20   30 Epochs     0    10   20   30 Epochs   0    10   20   30

  Smooth convergence         Oscillating             Loss explosion
  Stable learning            Some instability         then stabilizes low
  OPTIMAL ✓                  Acceptable               Too high ✗
```

---

## Results & Model Comparison

### Performance Summary

```
┌──────────────────────────────────────────────────────────────┐
│                    Model Performance Comparison               │
└──────────────────────────────────────────────────────────────┘

Model                    │ Embedding    │ Context    │ Performance
─────────────────────────┼──────────────┼────────────┼─────────────
Option 1: MLP Random     │ 15-dim       │ Bigram     │ Moderate
                         │ (learned)    │ (2 words)  │
─────────────────────────┼──────────────┼────────────┼─────────────
Option 2: MLP + Word2Vec │ 300-dim      │ Bigram     │ ⭐ BEST
                         │ (pre-trained)│ (2 words)  │
─────────────────────────┼──────────────┼────────────┼─────────────
Option 3: BiLSTM-MEMM    │ 15-dim       │ Full       │ Lower
                         │ (learned)    │ sentence   │ (tuning issues)
```

### Why Option 2 Performed Best

```
Advantages of Pre-trained Embeddings (Option 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Small Dataset + Pre-trained = Better Generalization
                Embeddings

┌──────────────────┐
│  Word2Vec Model  │  Trained on billions of words
│  (Google News)   │  from Google News corpus
└────────┬─────────┘
         │
         │ Rich semantic representations
         │ ("love" ≈ "enjoy", "great")
         ▼
┌──────────────────┐
│  Small Training  │
│  Dataset         │  Only thousands of sentences
└────────┬─────────┘
         │
         │ Fine-tune classifier, not embeddings
         ▼
┌──────────────────┐
│ Better           │  Leverage world knowledge
│ Performance      │  Less overfitting
└──────────────────┘


Random Embeddings (Option 1):
• Must learn word meanings from scratch
• Limited training data
• May overfit or underfit

BiLSTM (Option 3):
• More parameters to train
• Requires more data for optimal performance
• Complex architecture needs careful tuning
```

### Evaluation Metrics

```
Evaluation Metrics for Sequence Tagging
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prediction vs Ground Truth:

Predicted:  [T-POS, T-POS, O,     T-NEG, O    ]
True:       [T-POS, O,     T-POS, T-NEG, O    ]
            ──────────────────────────────────
            ✓      ✗      ✗      ✓      ✓

Metrics Computed:
─────────────────

True Positives (TP):
  Predicted sentiment tag (not O) AND correct

False Positives (FP):
  Predicted sentiment tag but was O, or wrong sentiment

False Negatives (FN):
  Predicted O but should be sentiment tag

Precision = TP / (TP + FP)
  → Of predicted sentiments, how many were correct?

Recall = TP / (TP + FN)
  → Of actual sentiments, how many did we find?

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
  → Harmonic mean, balances precision and recall
```

---

## Installation & Usage

### Requirements

```bash
# Python 3.7+
pip install torch
pip install numpy
pip install gensim
pip install scikit-learn
pip install tqdm
pip install matplotlib
```

### Data Requirements

```
Expected Data Files:
━━━━━━━━━━━━━━━━━━
dmemm/
├── train_set.pkl      # Training data (pickled)
├── test_set.pkl       # Test data (pickled)
└── GoogleNews-vectors-negative300.bin  # Word2Vec (for Option 2)

Data Format:
Each pickle file contains a list of dictionaries:
[
    {
        'words': ['word1', 'word2', ...],
        'ts_raw_tags': ['T-POS', 'O', ...]
    },
    ...
]
```

### Running the Models

#### Option 1: MLP with Random Embeddings

```bash
cd dmemm
python mlp.py
```

**Key Parameters** (edit in file):
- `EMBEDDING_DIM = 15` - Dimension of learned embeddings
- `CONTEXT_SIZE = 3` - Size of n-gram context
- `num_epochs = 15` - Number of training epochs
- `learning_rate = 0.001` - Adam optimizer learning rate

#### Option 2: MLP with Word2Vec

```bash
cd dmemm

# Download Word2Vec model first:
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# Place GoogleNews-vectors-negative300.bin in dmemm/

python mlp-word2vec.py
```

**Key Parameters**:
- Uses 300-dim Word2Vec embeddings (fixed)
- Loads first 50,000 words from Word2Vec
- Out-of-vocabulary words → zero vectors

#### Option 3: BiLSTM-MEMM

```bash
cd dmemm

# Training mode:
python bilstm.py --load_model 0

# Evaluation mode (load saved model):
python bilstm.py --load_model 1
```

**Key Parameters**:
- `EMBEDDING_DIM = 15` - Dimension of learned embeddings
- `HIDDEN_DIM = 10` - LSTM hidden size (5 per direction)
- `--load_model` - 0 for training, 1 to load saved model

### Model Outputs

```
Training Output:
━━━━━━━━━━━━━━━
• Loss curves per epoch
• Training progress with tqdm
• Final evaluation metrics (TP, FP, FN)
• Precision, Recall, F1 Score

Example:
Epoch 1/15: 100%|███████████| 45678/45678 [02:34<00:00]
Loss: 85.42
...
Epoch 15/15: 100%|██████████| 45678/45678 [02:31<00:00]
Loss: 28.15

Evaluation:
tp, fp, fn: 1234, 567, 234
Precision: 0.685
Recall: 0.841
F1: 0.755
```

---

## Project Structure

```
dmemm/
│
├── README.md                               # This file
│
├── dmemm/
│   ├── mlp.py                              # Option 1: Random embeddings
│   ├── mlp-word2vec.py                     # Option 2: Word2Vec embeddings
│   ├── bilstm.py                           # Option 3: BiLSTM-MEMM
│   ├── report.pdf                          # Detailed experimental results
│   │
│   ├── train_set.pkl                       # Training data (required)
│   ├── test_set.pkl                        # Test data (required)
│   └── GoogleNews-vectors-negative300.bin  # Word2Vec model (required for Option 2)
│
└── saved_models/                           # (created during training)
    └── hw2-bilstm.pt                       # Saved BiLSTM model
```

---

## Key Insights

### 1. Embeddings Matter for Small Datasets

```
Random vs Pre-trained Embeddings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Small Dataset Scenario:
┌──────────────────────────────────────────────┐
│                                              │
│  Random Embeddings:                          │
│  • Must learn "love" means positive          │
│  • Needs many examples                       │
│  • May not generalize well                   │
│                                              │
│  Pre-trained Embeddings:                     │
│  • Already knows "love" ≈ "enjoy" ≈ "great" │
│  • Semantic knowledge from billions of words │
│  • Better generalization ✓                   │
│                                              │
└──────────────────────────────────────────────┘
```

### 2. Context Window Trade-offs

```
Bigram (Options 1 & 2) vs Full Sentence (Option 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bigram Context:
  "I [love this] movie"
        └─┬─┘
    2-word window

  Pros: Simple, fewer parameters, faster
  Cons: Limited context

BiLSTM Context:
  "[I love this movie]"
   └──────┬──────┘
   Full sentence

  Pros: Long-range dependencies, full context
  Cons: More parameters, needs more data, slower
```

### 3. MEMM Sequential Dependencies

```
Why Model Previous Tags?
━━━━━━━━━━━━━━━━━━━━━━

Sentence: "love this movie but hate ending"

Without Previous Tag:
  love → T-POS
  this → ?
  movie → ?
  but → ?
  hate → T-NEG

With Previous Tag (MEMM):
  love → T-POS
  this → T-POS (likely continues positive)
  movie → T-POS (still in positive phrase)
  but → O (transition word)
  hate → T-NEG (negative)

MEMMs capture:
• Sentiment tends to span multiple words
• Transition patterns (POS → POS more likely than POS → NEG → POS)
• Sequential structure of language
```

### 4. Viterbi Decoding (Option 3)

```
Greedy vs Viterbi
━━━━━━━━━━━━━━━━━

Greedy (Options 1 & 2):
  Predict each tag independently
  → May produce inconsistent sequences

Viterbi (Option 3):
  Find globally optimal sequence
  → Consistent, respects transition probabilities

Example:
Greedy:  [T-POS, O, T-POS, T-NEG, O, T-POS]
         (inconsistent, jumpy)

Viterbi: [T-POS, T-POS, T-POS, O, T-NEG, T-NEG]
         (smooth transitions, more realistic)
```

### 5. Hyperparameter Tuning Importance

From the experimental results:

```
Learning Rate Impact
━━━━━━━━━━━━━━━━━━━

Too Low (< 0.001):
  ⊙ Slow convergence
  ⊙ May not reach optimal

Optimal (0.001):
  ⊙ Smooth convergence ✓
  ⊙ Stable training ✓
  ⊙ Best performance ✓

Too High (> 0.01):
  ⊙ Oscillating loss
  ⊙ May miss optimal
  ⊙ Can explode

Always tune:
• Learning rate
• Batch size
• Network architecture
• Embedding dimensions
• Number of epochs
```

---

## Future Improvements

```
Potential Enhancements
━━━━━━━━━━━━━━━━━━━━━

1. Contextualized Embeddings
   ├─ Replace Word2Vec with BERT/RoBERTa
   └─ Dynamic representations per context

2. CRF Layer (instead of MEMM)
   ├─ Conditional Random Fields
   └─ Model global sequence dependencies

3. Attention Mechanisms
   ├─ Weighted context aggregation
   └─ Interpretable focus on important words

4. Data Augmentation
   ├─ Synonym replacement
   ├─ Back-translation
   └─ Increase training data size

5. Ensemble Methods
   ├─ Combine all three options
   └─ Voting or stacking

6. Multi-task Learning
   ├─ Joint training on related tasks
   └─ Transfer learning from larger datasets
```

---

## References

- **Maximum Entropy Markov Models**: McCallum et al. (2000)
- **Word2Vec**: Mikolov et al. (2013) - "Efficient Estimation of Word Representations"
- **BiLSTM for Sequence Tagging**: Graves & Schmidhuber (2005)
- **Viterbi Algorithm**: Viterbi (1967)
- **PyTorch**: https://pytorch.org/

---

## License

Academic project for CS 577 - Natural Language Processing

---

## Author

Joshua Yeung

For questions or issues, please refer to the code documentation or the detailed report.pdf.

---

## Appendix: Mathematical Formulation

### MEMM Probability

```
P(tag_sequence | word_sequence) = ∏ P(tag_i | tag_{i-1}, word_i, context_i)
                                   i=1

where each local probability is modeled by a neural network:

P(tag_i | features) = exp(NN(features)_i) / Σ exp(NN(features)_j)
                                            j
                    = softmax(NN(features))_i
```

### BiLSTM Forward Equations

```
Forward LSTM:
→h_t = LSTM_forward(embedding_t, →h_{t-1})

Backward LSTM:
←h_t = LSTM_backward(embedding_t, ←h_{t+1})

Combined:
h_t = [→h_t ; ←h_t]  (concatenation)

Features:
f_t = W × h_t + b    (linear projection to tag space)
```

### Viterbi Dynamic Programming

```
Initialization:
π_0(START) = 0
π_0(tag) = -∞ for tag ≠ START

Recursion:
π_t(tag) = max[π_{t-1}(prev_tag) + score(prev_tag → tag) + feature(word_t, tag)]
           prev_tag

Backtracking:
best_tag_T = argmax π_T(tag)
             tag
Trace back through saved pointers to find optimal sequence
```

---

**Happy Training!** 🚀

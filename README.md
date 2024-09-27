# Wine Reviews Transformer Model

This project demonstrates the Transformer architecture with a specific focus on how the attention mechanism works in natural language processing (NLP). The dataset used consists of approximately 129,971 wine reviews, capturing detailed descriptions of wines from various countries and regions around the world.

## Table of Contents
- [Project Overview](#project-overview)
- [Attention Mechanism](#attention-mechanism)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results and Insights](#results-and-insights)
- [License](#license)

## Project Overview

The primary goal of this project is to showcase the transformer architecture, widely known for its ability to handle sequence data through attention rather than recurrence. This project demonstrates how transformers can be applied to text generation and language modeling using a dataset of wine reviews.

## Attention Mechanism

The attention mechanism is central to how the transformer processes data. This project aims to provide a visual and interactive demonstration of how attention works in predicting the next word in a sequence. Specifically, this model learns to predict the next word in a wine review, based on prior words in the sequence.

The transformer model incorporates a causal attention mask, which ensures that the model can only attend to past or current tokens, making it ideal for autoregressive tasks like next-word prediction.

## Dataset

The dataset contains 129,971 wine reviews, including information on:

- Country of origin
- Province (region) of the wine
- Wine variety (grape type)
- The wine description
- price
- designation

**Dataset download link:**
```
https://www.kaggle.com/datasets/zynicide/wine-reviews?select=winemag-data-130k-v2.json
```

**Download dataset using kaggle api:**
```
kaggle datasets download -d zynicide/wine-reviews
```


Each review is formatted into a structured sequence that is fed into the transformer for training.

Example of a wine review:

``
"wine review : France : Bordeaux : Merlot : This is a full-bodied wine with deep fruity flavors and a hint of oak."
``


## Model Architecture

The transformer architecture used in this project is composed of the following components:

- **Token and Positional Embedding**: Converts text into numerical vectors for input.
- **Multi-Head Self-Attention**: Enables the model to focus on different parts of the sequence simultaneously.
- **TextGenerator**: Generates text according to the token with highest likelihood or attention.
- **Causal Attention Mask**: Ensures that the model only attends to earlier tokens in the sequence for autoregressive training.

### Hyperparameters:
- **VOCAB_SIZE:** 10,000 (limits the vocabulary to the most frequent words)
- **MAX_LEN:** 80 (maximum length of input sequences)
- **EMBEDDING_DIM:** 256 (size of word embeddings)
- **K_DIM:** 256 (dimension of key vectors in attention)
- **N_HEADS:** 2 (number of attention heads)
- **FF_DIM:** 256 (dimension of feed-forward layers)
- **BATCH_SIZE:** 64
- **EPOCHS:** 15

## Training

The training process involves tokenizing each wine review and creating a shifted version of the sequence for next-word prediction. The transformer model is trained to predict the next token based on prior tokens in the review.

### Training Process:
1. **Text Vectorization:** Tokenizes the wine reviews into sequences of integers.
2. **Causal Masking:** Ensures that the model only attends to previous tokens.
3. **Training:** The model is trained on the tokenized sequences, optimizing for the next word prediction.

## Requirements

- Python 3.12+
- TensorFlow 2.17+
- NumPy
- Matplotlib
- keras

You can install the required packages via:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wine-reviews-transformer.git
   cd wine-reviews-transformer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train.py
   ```

4. Visualize the attention mechanism:
   Once the model is trained, the attention weights can be visualized using the provided visualization tools.

## Results and Insights

This project demonstrates how transformers can be used for sequence modeling and next-word prediction in the context of natural language. The attention mechanism allows the model to focus on key parts of a wine review when predicting the next word, offering insights into how transformers handle text generation tasks.

Example Output:
```
Input: "wine review : France : Bordeaux : Merlot :"
Predicted next word: "rich"
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

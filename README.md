# Sequence-to-Sequence Neural Machine Translation

A Python implementation of a Sequence-to-Sequence (Seq2Seq) model for English to French translation using PyTorch.

## Project Overview

This project implements a basic Seq2Seq model with:
- Encoder-Decoder architecture using LSTM layers
- Word embeddings for text representation
- Teacher forcing mechanism for training
- Batch processing capabilities

## Model Architecture

### Encoder
- Embedding layer for input sequences
- LSTM layers for processing English sentences
- Returns hidden states and cell states for decoder initialization

### Decoder
- Takes encoder states as initial states
- Generates French translations word by word
- Uses teacher forcing during training
- Final linear layer for vocabulary prediction

## Requirements

```bash
torch
numpy
pandas
```

## Usage

1. Prepare your data in CSV format with English-French pairs
2. Update hyperparameters in `train.py` if needed:
```python
embedding_size = 200
hidden_size = 200
num_layers = 2
dropout = 0.5
```
3. Run training:
```bash
python train.py
```

## File Structure

- `model.py`: Contains Encoder, Decoder, and Seq2Seq model implementations
- `train.py`: Training script and hyperparameters
- `data.py`: Data loading and preprocessing utilities

## Training Results

The model was trained for 14 epochs with the following key observations:

### Training Progress
- Started with initial loss: 1.9379 (Epoch 1)
- Final loss: 1.4777 (Epoch 14)
- Consistent improvement in loss across epochs
- Best model saved at each improvement

## Future Improvements

- Add attention mechanism
- Implement beam search for inference
- Add data augmentation
- Improve translation accuracy
- Add more diverse test cases

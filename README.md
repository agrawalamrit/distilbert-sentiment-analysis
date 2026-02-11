# DistilBERT Sentiment Analysis

Fine-tuned DistilBERT on the IMDB dataset (50,000 labeled reviews) for binary sentiment classification.

## Model
- Base Model: distilbert-base-uncased
- Framework: HuggingFace Transformers
- Backend: PyTorch
- Device: MPS (Apple Silicon GPU acceleration)

## Dataset
- IMDB Movie Reviews Dataset
- 25,000 training samples
- 25,000 test samples

## Training
- Epochs: 1
- Batch Size: 8
- Evaluation Strategy: Per Epoch

## Results
Test Accuracy: XX% (replace with actual value)

## Run Training
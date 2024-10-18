import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import pipeline

# Specify the model name
model_name = 'albert-base-v2'

# Load the tokenizer
tokenizer = AlbertTokenizer.from_pretrained(model_name)

# Load the pre-trained ALBERT model for sequence classification
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification

# Sample texts for classification
texts = [
    "I absolutely loved the new Batman movie!",
    "The food at the restaurant was terrible and the service was slow."
]

# Tokenize the input texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    outputs = model(**inputs)

# The outputs are logits; apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the predicted class labels
predictions = torch.argmax(probabilities, dim=-1)

# Map predictions to labels (assuming 0: Negative, 1: Positive)
label_mapping = {0: "Negative", 1: "Positive"}

# Display the results
for text, pred, prob in zip(texts, predictions, probabilities):
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {label_mapping[pred.item()]}")
    print(f"Probability: {prob[pred].item():.4f}\n")


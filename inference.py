from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_path = "out/checkpoint-3125"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

while True:
    text = input("Enter a sentence (or type exit): ")
    if text.lower() == "exit":
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    label = "Positive" if prediction.item() == 1 else "Negative"

    print(f"\nSentiment: {label}")
    print(f"Confidence: {confidence.item():.4f}\n")

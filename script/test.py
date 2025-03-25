from transformers import BertForSequenceClassification, BertTokenizer
import json
import torch
import torch.nn.functional as F

model = BertForSequenceClassification.from_pretrained("./init_model")
tokenizer = BertTokenizer.from_pretrained("./init_model")
model.eval()

with open("data/test_dataset.json", "r", encoding="utf-8") as file:
    test_examples = json.load(file)

correct = 0
total = len(test_examples)
results = []
incorrect_cases = []

for idx, example in enumerate(test_examples, start=1):
    inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)[0]
    prediction = probabilities.argmax().item()
    confidence = probabilities[prediction].item() * 100  

    if prediction == example["labels"]:
        correct += 1
    else:
        incorrect_cases.append({
            "line_number": idx,
            "text": example["text"],
            "predicted_labels": prediction,
            "true_labels": example["labels"],
            "confidence_percent": round(confidence, 2)
        })

accuracy = correct / total * 100
results.append({
        "accuracy_percent": round(accuracy, 2),
        "total_cases_num": total,
        "incorrect_cases_num": len(incorrect_cases),
        "incorrect_cases": incorrect_cases
        }) 
print(f"Accuracy: {accuracy:.2f}%")

with open("data/result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

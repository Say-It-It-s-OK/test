from transformers import BertForSequenceClassification, BertTokenizer
import torch

model = BertForSequenceClassification.from_pretrained("./init_model")
tokenizer = BertTokenizer.from_pretrained("./init_model")
model.eval()

input_text = "메뉴판 보여줘"
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=-1).item()
if prediction == 0:
    print("Query 요청입니다.")
else:
    print("Order 요청입니다.")
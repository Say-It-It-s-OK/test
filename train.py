from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import json

def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def preprocess_data(example):
    return tokenizer(example['text'], truncation=True, padding=True, max_length=128)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

query_data = load_json_data("query_dataset.json")
query_dataset = Dataset.from_dict({
    "text": [item["text"] for item in query_data],
    "label": [0] * len(query_data)
})

order_data = load_json_data("order_dataset.json")
order_dataset = Dataset.from_dict({
    "text": [item["text"] for item in order_data],
    "label": [1] * len(order_data)
})

dataset = concatenate_datasets([query_dataset, order_dataset])
dataset = dataset.map(preprocess_data, batched=True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',          # 출력 디렉토리
    num_train_epochs=10,              # 훈련 에폭 수
    per_device_train_batch_size=8,   # 배치 크기
    per_device_eval_batch_size=8,    # 평가 배치 크기
    warmup_steps=500,                # 워밍업 단계 수
    weight_decay=0.01,               # 가중치 감소
    logging_dir='./logs',            # 로그 저장 디렉토리
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # 학습할 모델
    args=training_args,                  # 훈련 인자
    train_dataset=dataset,               # 훈련 데이터셋
    eval_dataset=dataset,                # 평가 데이터셋
)

trainer.train()
trainer.evaluate()

model.save_pretrained("./init_model")
tokenizer.save_pretrained("./init_model")
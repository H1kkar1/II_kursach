import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets.config_dialog import BOT_CONFIG
# Создаем датасет: каждому примеру соответствует один ответ
X = []
y = []
for intent_name, intent_data in BOT_CONFIG["intents"].items():
    examples = intent_data["examples"]
    responses = intent_data["responses"]

    # Сопоставляем каждый пример с каждым ответом (можно изменить логику)
    for example in examples:
        for response in responses:
            X.append(example)
            y.append(response)

# Кодируем ответы в числовые метки
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

os.makedirs("../model_cache/bert/", exist_ok=True)
with open("../model_cache/bert/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
# Разделяем на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# 2. Создание датасета
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 3. Инициализация модели и токенизатора
MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer.save_pretrained("../model_cache/bert/")
# Создаем DataLoader
train_dataset = IntentDataset(X_train, y_train, tokenizer)
test_dataset = IntentDataset(X_test, y_test, tokenizer)

batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size
)

# 4. Обучение модели
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3 эпохи
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_values = []
accuracy_values = []

for epoch in range(3):
    print(f'Epoch {epoch + 1}/{3}')
    print('-' * 10)

    # Обучение
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct_predictions.double() / len(train_dataset)

    loss_values.append(avg_loss)
    accuracy_values.append(avg_acc)

    print(f'Train loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

        # Валидация
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            val_correct += torch.sum(preds == labels)

    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = val_correct.double() / len(test_dataset)

    print(f'Validation loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}\n')

# 5. Визуализация результатов
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_values, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_values, label='Train Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
model.save_pretrained("../model_cache/bert/")
print("Модель и артефакты сохранены в папку '../model_cache/bert/'")
# 6. Функция для генерации ответа


def get_response(text, model, tokenizer, label_encoder, threshold=0.7):
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    if confidence < threshold:
        return "Извините, я не уверен в ответе. Можете уточнить вопрос?"

    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    return predicted_label


# Пример использования
test_questions = [
    "Где можно купить билет на поезд?",
    "Какие есть официальные сайты?",
    "Какой сервис самый надежный?"
]

for question in test_questions:
    response = get_response(question, model, tokenizer, label_encoder)
    print(f"Вопрос: {question}")
    print(f"Ответ: {response}\n")
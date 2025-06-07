import os

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split
from config_dialog import BOT_CONFIG

# Создаем пары "вход-выход" для обучения
data_pairs = []
for intent_name, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        for response in intent_data['responses']:
            data_pairs.append(f"Вход: {example}\nВыход: {response}")

# Разделяем данные на обучающую и валидационную выборки
train_data, val_data = train_test_split(data_pairs, test_size=0.2, random_state=42)


# 2. Создание класса Dataset
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


# 3. Загрузка модели и токенизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_path = "../model_cache/models--sberbank-ai--rugpt3large_based_on_gpt2/snapshots"
snapshot = sorted(os.listdir(cache_path))[-1]  # Берём последнюю версию
model_path = f"{cache_path}/{snapshot}"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token

class CustomGPT2Model(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(model_path)

        # Замораживаем только первые 6 слоёв
        for layer in self.gpt.transformer.h[:-6]:
            for param in layer.parameters():
                param.requires_grad = False

        # Добавляем обработку новых токенов
        self.gpt.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

model = CustomGPT2Model(model_path).to(device)
model.gpt.resize_token_embeddings(len(tokenizer))

# 4. Подготовка данных для обучения
train_dataset = ConversationDataset(tokenizer, train_data)
val_dataset = ConversationDataset(tokenizer, val_data)

# 5. Настройка обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False,
)

# 6. Создание Trainer и обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 7. Сохранение модели
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')


# 8. Пример использования обученной модели
def generate_response(prompt, model, tokenizer, max_length=128):
    input_text = f"Вход: {prompt}\nВыход:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Пример
prompt = "Привет, насчет билетов"
print(generate_response(prompt, model, tokenizer))
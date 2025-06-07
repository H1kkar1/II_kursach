import torch
from torch import nn
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datasets.config_dialog2 import BOT_CONFIG
import os
import logging
# Отключение несущественных предупреждени
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Настройка логирования

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 1. Инициализация модели и токенизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cache_path = "./model_cache/models--sberbank-ai--rugpt3large_based_on_gpt2/snapshots"
# snapshot = sorted(os.listdir(cache_path))[-1]  # Берём последнюю версию
# model_path = f"{cache_path}/{snapshot}"
model_path = "./best_gpt_model_2"
logger.info("Инициализация модели и токенизатора...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer.pad_token = tokenizer.eos_token
print("Модель загружена:", model is not None)
input_text = "Привет"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
logger.info(f"Модель загружена на устройство: {device}")

class CustomGPT2Model(nn.Module):
    def __init__(self, model_name):
        super(CustomGPT2Model, self).__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name)

        # Пример добавления дополнительного линейного слоя
        self.additional_layer = nn.Linear(self.gpt.config.hidden_size, self.gpt.config.hidden_size)

        # Пример добавления dropout слоя
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Получаем выходы из оригинальной модели
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Применяем дополнительные слои
        hidden_states = outputs.hidden_states[-1]  # Получаем скрытые состояния из последнего слоя
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.additional_layer(hidden_states)

        # Возвращаем обновленные выходы
        return (hidden_states, ) + outputs[1:]

# Пример использования
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
custom_model = CustomGPT2Model(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Пример генерации текста с использованием кастомной модели
input_text = "Привет"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = custom_model.generate(**inputs)
print(tokenizer.decode(outputs[0]))

def generate_response(user_input, model, tokenizer, max_length=100):
    prompt = f"Пользователь: {user_input}\nБот:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )

    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Полный ответ: '{full_response}'")  # Добавлено для отладки
    # Извлекаем только ответ бота
    bot_response = full_response.split("Бот:")[1].strip() if "Бот:" in full_response else full_response
    return bot_response

# 2. Подготовка данных
train_texts = []
test_texts = []

for intent_name, intent_data in BOT_CONFIG["intents"].items():
    examples = intent_data["examples"]
    responses = intent_data["responses"]

    if len(examples) >= 1:  # Изменено условие на >= 1
        split_idx = max(1, int(0.8 * len(examples)))
        logger.info(f"Split index: {split_idx}")

        for example, response in zip(examples[:split_idx], responses[:split_idx]):
            train_texts.append(f"Пользователь: {example}\nБот: {response}")

        for example, response in zip(examples[split_idx:], responses[split_idx:]):
            test_texts.append((example, response))
    else:
        for example, response in zip(examples, responses):
            train_texts.append(f"Пользователь: {example}\nБот: {response}")

logger.info(f"Total train texts: {len(train_texts)}")
logger.info(f"Total test texts: {len(test_texts)}")
# Функция для оценки точности

def evaluate_accuracy(model, tokenizer, test_data, num_samples=3):
    if not test_data:
        logger.warning("Тестовая выборка пуста, точность не может быть оценена")
        return 0.0

    sampled_data = test_data[:num_samples]
    correct = 0

    for user_input, expected_response in sampled_data:
        try:
            generated_response = generate_response(user_input, model, tokenizer)
            logger.info(f"Сгенерированный ответ: '{generated_response}'")
            logger.info(f"Ожидаемый ответ: '{expected_response}'")

            # Простая оценка: считаем правильным, если ожидаемый ответ содержится в сгенерированном
            if expected_response.lower() in generated_response.lower():
                correct += 1
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа для '{user_input}': {str(e)}")
            continue

    accuracy = correct / num_samples if num_samples > 0 else 0.0
    return accuracy



# Оценка точности до обучения
if test_texts:
    initial_accuracy = evaluate_accuracy(model, tokenizer, test_texts)
    logger.info(f"Точность до обучения: {initial_accuracy:.2%}")
else:
    initial_accuracy = 0.0
    logger.warning("Недостаточно данных для оценки точности до обучения")

def log_generation_example(prompt, model, tokenizer):
    logger.info(f"\nТест генерации ДО обучения:")
    logger.info(f"Ввод: '{prompt}'")
    output = generate_response(prompt, model, tokenizer)
    logger.info(f"Вывод: '{output}'")


# 3. Датасет
class GPTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bot_token = tokenizer.encode("\nБот:")[0]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Маска для обучения только на ответах бота
        labels = input_ids.clone()
        try:
            bot_pos = (input_ids == self.bot_token).nonzero()[0].item()
            labels[:bot_pos] = -100  # Игнорируем в loss
        except IndexError:
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# 4. Создание DataLoader
train_dataset = GPTDataset(train_texts, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Меньший batch_size для GPT-3

# 5. Оптимизатор и планировщик
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
total_steps = len(train_loader) * 3  # 3 эпохи
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# 6. Обучение
train_losses = []
train_accuracies = [initial_accuracy]
print(train_accuracies) # Начинаем с начальной точности
best_loss = float('inf')
logger.info("\nНачало обучения...")

for epoch in range(3):  # 3 эпохи
    model.train()
    epoch_loss = 0
    total_batches = len(train_loader)

    # Инициализация progress_bar без batch_idx в цикле
    progress_bar = tqdm(enumerate(train_loader), total=total_batches, desc=f'Epoch {epoch + 1}')

    for batch_idx, batch in progress_bar:
        optimizer.zero_grad()

        # Перенос данных на устройство
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device)
        }

        # Прямой проход
        outputs = model(**inputs)
        loss = outputs.loss

        # Обратное распространение
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Логирование и обновление прогресса
        epoch_loss += loss.item()
        current_loss = loss.item()
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        # Логирование каждые 10 батчей
        if batch_idx % 10 == 0:
            logger.info(f'Batch {batch_idx}/{total_batches}, Loss: {current_loss:.4f}')

    # После эпохи
    avg_loss = epoch_loss / total_batches
    train_losses.append(avg_loss)

    # Оценка точности после эпохи
    if test_texts:
        current_accuracy = evaluate_accuracy(model, tokenizer, test_texts)
        train_accuracies.append(current_accuracy)
        logger.info(f"Эпоха {epoch + 1} завершена. Средний Loss: {avg_loss:.4f}, Точность: {current_accuracy:.2%}")
    else:
        train_accuracies.append(0.0)
        logger.info(f"Эпоха {epoch + 1} завершена. Средний Loss: {avg_loss:.4f}")
    print(train_accuracies)
    # Тестовая генерация
    test_prompt = "Как купить билет?"
    logger.info(f"\nТест генерации после эпохи {epoch + 1}:")
    logger.info(f"Ввод: '{test_prompt}'")
    generated_text = generate_response(test_prompt, model, tokenizer)
    logger.info(f"Вывод: '{generated_text}'")

    # Сохранение лучшей модели
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save_pretrained("best_gpt_model_2_1")
        tokenizer.save_pretrained("best_gpt_model_2_1")
        logger.info(f"Сохранена лучшая модель с loss {best_loss:.4f}")

        # Дополнительно сохраняем полное состояние для возобновления обучения
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
        }, 'best_model_checkpoint.pt')

# Оценка точности после обучения
if test_texts:
    final_accuracy = evaluate_accuracy(model, tokenizer, test_texts)
    logger.info(f"Точность после обучения: {final_accuracy:.2%}")
else:
    final_accuracy = 0.0
    logger.warning("Недостаточно данных для оценки точности после обучения")

# 7. Визуализация обучения
plt.figure(figsize=(12, 5))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# График точности
plt.subplot(1, 2, 2)
epochs = range(len(train_accuracies))
plt.plot(epochs, train_accuracies, label='Accuracy', color='green', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
if initial_accuracy > 0:  # Исправлено: используем initial_accuracy вместо initial_quality
    plt.axhline(y=initial_accuracy, color='r', linestyle='--', label='Initial Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Сохранение финальной модели
model.save_pretrained("best_gpt_model_2_1")
tokenizer.save_pretrained("best_gpt_model_2_1")

loaded_model = GPT2LMHeadModel.from_pretrained("best_gpt_model_2_1").to(device)
loaded_tokenizer = GPT2Tokenizer.from_pretrained("best_gpt_model_2_1")

# Тестирование
test_phrases = ["Привет", "Какие поезда идут в Москву?", "Помоги купить билет"]
for phrase in test_phrases:
    response = generate_response(phrase, loaded_model, loaded_tokenizer)
    print(f"Пользователь: {phrase}")
    print(f"Бот: {response}\n")
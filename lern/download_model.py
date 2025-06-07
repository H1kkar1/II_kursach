from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

MODEL_NAME = "sberbank-ai/rugpt3large_based_on_gpt2"
CACHE_DIR = "./model_cache"  # Локальная папка для сохранения

# Создаем папку, если ее нет
os.makedirs(CACHE_DIR, exist_ok=True)

print("Скачивание токенизатора...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

print("Скачивание модели...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

print(f"Модель и токенизатор сохранены в: {os.path.abspath(CACHE_DIR)}")

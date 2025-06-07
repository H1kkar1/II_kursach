import random
from datetime import datetime

import numpy as np
import torch
import os
import warnings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
# Отключение несущественных предупреждений
from datasets.config_dialog import BOT_CONFIG
from lern.intent_classifier import IntentClassifier

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_work.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChatNeural:
    def __init__(self, model_dir="best_gpt_model_2_1"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_generator()
        self.last_question = ""
        self.classifier = IntentClassifier()

    def _init_generator(self):
        """Инициализация генеративной модели"""
        try:
            # Загрузка сохраненной модели
            if os.path.exists(self.model_dir):
                self.generator_tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
                self.generator_model = GPT2LMHeadModel.from_pretrained(self.model_dir).to(self.device)
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
                print("Успешно загружена обученная GPT-модель")
            else:
                raise FileNotFoundError(f"Модель не найдена в {self.model_dir}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")
        return True

    def _postprocess_generated(self, text):
        """Улучшенная очистка ответа от всех следов бота и билетов"""
        import re
        logger.info(f"Начало пост-обработки")
        # 1. Удаление временных меток и скобок
        text = re.sub(r'(Бот|Консультант)\s*\(?\d{1,2}:\d{2}\)?:?', '', text, flags=re.IGNORECASE)

        # 2. Удаление всех вариантов подписей бота
        bot_prefixes = [
            r'Бот:\s*',
            r'Консультант:\s*',
            r'Ответ:\s*',
            r'➡\s*',
            r'\[Бот\]:?',
            r'\(\s*бот\s*\):?'
        ]
        for prefix in bot_prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)

        # 3. Удаление всего после последнего знака препинания если дальше идет подпись
        text = re.split(r'(?<=[.!?])\s*(?=[А-ЯA-Z])', text)[0]

        # 5. Фильтрация повторяющихся фраз
        sentences = [sent.strip() for sent in re.split(r'(?<=[.!?])\s+', text) if sent.strip()]
        unique_sentences = []
        for sent in sentences:
            if not any(sent.lower() == unique.lower() for unique in unique_sentences):
                unique_sentences.append(sent)

        # 6. Выбор не более 2 осмысленных предложений
        final_text = ' '.join(unique_sentences[:2]).strip()

        # 7. Фильтр "пустых" ответов
        if len(final_text.split()) < 2 or final_text.lower().startswith('бот'):
            return "Пожалуйста, уточните ваш вопрос."

        return final_text

    def get_sklern_response(self, user_input: str,  config: dict = BOT_CONFIG) -> str:
        """
        Получает ответ бота на основе предсказанного намерения

        Args:
            user_input (str): Входной текст пользователя
            classifier (IntentClassifier): Обученный классификатор намерений
            config (dict): Конфигурация бота (BOT_CONFIG)

        Returns:
            str: Ответ бота
        """
        # Предсказываем намерение
        intent, confidence = self.classifier.predict(user_input)

        # Если уверенность низкая, возвращаем стандартный ответ
        if confidence < 0.3:
            return "Извините, я не совсем понял ваш вопрос. Можете переформулировать?"

        # Получаем случайный ответ для этого намерения
        responses = config['intents'].get(intent, {}).get('responses', [])

        if not responses:
            return "Я пока не знаю ответа на этот вопрос"

        # Возвращаем случайный ответ из доступных
        return np.random.choice(responses)

    def _generate_response(self, text, max_length=50):
        """Генерация ответа с улучшенным контролем качества"""
        # 1. Подготовка промпта с четкими инструкциями
        prompt = (
            "Ты - консультант по ж/д билетам. Отвечай ТОЛЬКО на заданный вопрос. "
            "Соблюдай правила:\n"
            "1. Никогда не упоминай билеты, если о них не спрашивают\n"
            "2. Отвечай 1-2 предложениями\n"
            "3. Не используй фразы 'Бот:', 'Консультант:'\n"
            f"Вопрос: {text}\n"
            "Ответ:"
        )

        # 2. Подготовка входных данных
        inputs = self.generator_tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # 3. Параметры генерации с динамической адаптацией
        gen_params = {
            "max_new_tokens": min(max_length, 100),  # Жесткое ограничение длины
            "temperature": 0.5 if len(text.split()) < 8 else 0.7,  # Для коротких вопросов - строже
            "top_k": 40,
            "top_p": 0.85,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "num_beams": 2,  # Оптимально для скорости/качества
            "early_stopping": True,
            "pad_token_id": self.generator_tokenizer.eos_token_id
        }

        # 4. Генерация с двумя попытками
        for attempt in range(2):
            try:
                logger.info(f"генерация ответа на сообщение {text}")
                with torch.no_grad():
                    outputs = self.generator_model.generate(
                        **inputs,
                        **gen_params
                    )

                # 5. Декодирование и очистка ответа
                raw_response = self.generator_tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
                logger.info(F"ОТВЕТ: {raw_response}")
                # 6. Постобработка
                response = self._postprocess_generated(raw_response)
                return response

            except Exception as e:
                print(f"Ошибка генерации (попытка {attempt + 1}): {str(e)}")

        return "Не могу дать точный ответ. Уточните, пожалуйста, ваш вопрос."

    def get_response(self, text):
        if not text.strip():
            return "Я не расслышал ваш вопрос. Пожалуйста, повторите."

        # Сохраняем текущий вопрос перед обработкой
        self.last_question = text.lower()  # Сохраняем в нижнем регистре для удобства сравнения

        # 2. Генерация ответа моделью
        try:
            context = (
                f"Сегодня {datetime.now().strftime('%d.%m.%Y')}. "
                "Актуальные направления: Москва, Санкт-Петербург, Сочи, Казань.\n"
                f"Пользователь: {text}\n"
                "Консультант:"
            )

            generated = self._generate_response(context)
            return generated

        except Exception as e:
            print(f"Ошибка генерации: {str(e)}")
            return random.choice([
                "Уточните, пожалуйста, ваш вопрос.",
                "Для уточнения информации можете посетить наш сайт или позвонить по телефону 8-800-775-00-00."
            ])


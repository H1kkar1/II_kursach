from config_dialog import BOT_CONFIG
import random
from nlpaug import Augmenter
from nlpaug.augmenter.word import SynonymAug, ContextualWordEmbsAug
from transformers import pipeline
#
#
# generator = pipeline('text-generation', model='sberbank-ai/rugpt3small_based_on_gpt2')
# TARGET_SAMPLES = 150
# synonym_aug = SynonymAug(aug_src='wordnet', lang='rus')
# contextual_aug = ContextualWordEmbsAug(model_path='bert-base-multilingual-cased', action="insert")
#
#
# def generate_with_prompt(examples, num_samples=5):
#     """Генерация новых примеров по шаблону"""
#     prompt = f"Примеры запросов для интента:\n" + "\n".join(examples)
#     generated = generator(
#         prompt,
#         max_length=100,
#         num_return_sequences=num_samples,
#         temperature=0.7,
#         top_k=50
#     )
#     return [g['generated_text'].split('\n')[-1] for g in generated]
#
#
# def augment_text(text, num_aug=3):
#     """Генерирует вариации текста"""
#     augmented = []
#     try:
#         # Синонимичная замена
#         augmented += [synonym_aug.augment(text) for _ in range(num_aug)]
#
#         # Контекстуальное добавление слов
#         augmented += [contextual_aug.augment(text) for _ in range(num_aug)]
#
#         # Простые преобразования
#         words = text.split()
#         if len(words) > 3:
#             # Перестановка слов
#             shuffled = words.copy()
#             random.shuffle(shuffled)
#             augmented.append(' '.join(shuffled))
#
#             # Пропуск слов
#             augmented.append(' '.join(words[:-1]))
#         print("procces augment_text ")
#         return list(set(aug for aug in augmented if aug and aug != text))
#     except:
#         return []
#
#
# augmented_intents = {}
# for intent, data in BOT_CONFIG['intents'].items():
#     examples = data['examples']
#     current_count = len(examples)
#
#     if current_count >= TARGET_SAMPLES:
#         augmented_intents[intent] = examples
#         continue
#
#     needed = TARGET_SAMPLES - current_count
#     augmented = examples.copy()
#     print("augmented")
#     # Аугментация существующих примеров
#     for example in examples:
#         augmented += augment_text(example, num_aug=min(3, needed // len(examples) + 1))
#
#     # Догенерация недостающих
#     if len(augmented) < TARGET_SAMPLES:
#         try:
#             generated = generate_with_prompt(examples, num_samples=needed // 2)
#             augmented += generated
#         except:
#             pass
#
#     # Оставляем уникальные
#     augmented = list(set(augmented))[:TARGET_SAMPLES]
#     augmented_intents[intent] = augmented
#
# print("После аугментации:")
# for intent, examples in augmented_intents.items():
#     print(f"{intent}: {len(examples)} примеров")
#
# # Обновляем конфиг
# BOT_CONFIG['intents'] = {
#     intent: {'examples': examples, 'responses': BOT_CONFIG['intents'][intent]['responses']}
#     for intent, examples in augmented_intents.items()
# }
#
# # Сохраняем обновленный конфиг
# import json
# with open('augmented_config.json', 'w', encoding='utf-8') as f:
#     json.dump(BOT_CONFIG, f, ensure_ascii=False, indent=2)

import random

import random



import random
# Исходные примеры
examples = [
    "Где выгодно купить билеты на поезд Москва-Питер?",
    "Реклама жд билетов Москва-Сочи со скидкой",
    "Акция на билеты Якутия-Норильск!",
    "Специальное предложение: Москва-Санкт-Петербург от 999 руб!",
    "Рекламируем горящие билеты Москва-Сочи",
    "Как дешево добраться из Москвы в Питер на поезде?",
    "Уникальные цены на билеты Якутия-Норильск",
    "Реклама комфортных поездов Москва-Сочи",
    "Забронируйте билеты Москва-Питер онлайн со скидкой 20%",
    "Лучшие предложения на жд билеты Москва-Сочи",
    "Рекомендуем быстрые поезда Москва-Санкт-Петербург",
    "Эксклюзив: прямые рейсы Якутия-Норильск",
    "Распродажа билетов Москва-Питер в этом месяце",
    "Реклама ночных поездов Москва-Сочи",
    "Выгодные тарифы на маршруте Якутия-Норильск",
    "Только сейчас - скидки на билеты Москва-Питер!",
    "Реклама новых вагонов Москва-Сочи",
    "Удобное расписание поездов Якутия-Норильск",
    "Как сэкономить на билетах Москва-Санкт-Петербург?",
    "Рекламируем бизнес-класс Москва-Сочи",
    "Лучшие цены на билеты Якутия-Норильск в этом сезоне"
]
# Синонимы и вариации
synonyms = {
    'реклама': ['продвижение', 'рекламная акция', 'спецпредложение', 'акция'],
    'билеты': ['проездные', 'билеты на поезд', 'жд билеты', 'железнодорожные билеты'],
    'купить': ['приобрести', 'заказать', 'забронировать', 'оформить'],
    'поезд': ['состав', 'поезда', 'железнодорожный транспорт', 'рейс'],
    'скидка': ['выгода', 'спеццена', 'экономия', 'пониженная цена'],
    'маршрут': ['направление', 'путь', 'рейс', 'линия'],
    'предложение': ['вариант', 'оферта', 'возможность', 'условия'],
    'дешево': ['недорого', 'выгодно', 'по низкой цене', 'экономно'],
    'комфортный': ['удобный', 'современный', 'комфортабельный', 'премиальный'],
    'распродажа': ['акционные цены', 'специальные условия', 'скидочная кампания', 'промо']
}

def generate_examples(base_examples, num_variations):
    new_examples = set(base_examples)  # Используем set для уникальности
    for _ in range(num_variations):
        example = random.choice(base_examples)
        for word, variations in synonyms.items():
            if word in example:
                example = example.replace(word, random.choice(variations))
        new_examples.add(example)
    return list(new_examples)
# Генерация дополнительных примеров

new_examples = generate_examples(examples, 200)  # Генерируем 20 новых примеров
# Объединяем с исходными примерами
all_examples = list(set(examples) | set(new_examples))

print(all_examples)
import os
import re
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.config_dialog import BOT_CONFIG
from sklearn.linear_model import SGDClassifier

class IntentClassifier:
    def __init__(self, config: dict = BOT_CONFIG, model_dir: str = "models"):
        self.config = config
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "intent_classifier.joblib")
        self.metrics_path = os.path.join(self.model_dir, "training_metrics.png")
        self._prepare_data()
        self.train_accuracy = []
        self.val_accuracy = []
        self.train_loss = []

    def _prepare_data(self):
        """Подготовка данных для обучения из конфига"""
        self.texts = []
        self.labels = []

        for intent_name, intent_data in self.config['intents'].items():
            for example in intent_data['examples']:
                clean_text = self._clean_text(example)
                self.texts.append(clean_text)
                self.labels.append(intent_name)

    def train(self, test_size: float = 0.2, random_state: int = 42, n_epochs: int = 10):
        """Обучение модели с визуализацией метрик по эпохам"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.texts, self.labels,
            test_size=test_size,
            random_state=random_state
        )

        # Создаем пайплайн
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                max_features=10000,
                ngram_range=(1, 2),
            )),
            ('clf', SGDClassifier(
                loss='hinge',  # Используем hinge loss для имитации SVM
                penalty='l2',
                max_iter=100,
                tol=1e-3,
                random_state=random_state,
                warm_start=True
            ))
        ])

        # Подготовка для хранения метрик
        self.train_accuracy = []
        self.val_accuracy = []
        self.train_loss = []

        print("Starting training...")
        for epoch in range(n_epochs):
            # Частичное обучение (1 эпоха)
            if epoch == 0:
                self.model.fit(X_train, y_train)
            else:
                self.model.named_steps['clf'].max_iter += 100  # Увеличиваем итерации
                self.model.named_steps['clf'].fit(
                    self.model.named_steps['tfidf'].transform(X_train),
                    y_train
                )

            # Оценка качества
            train_pred = self.model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            self.train_accuracy.append(train_acc)

            val_pred = self.model.predict(X_test)
            val_acc = accuracy_score(y_test, val_pred)
            self.val_accuracy.append(val_acc)

            self.train_loss.append(1 - train_acc)

            print(f"Epoch {epoch + 1}/{n_epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Сохранение и визуализация
        self._save_model()
        self._plot_metrics(y_test, val_pred, n_epochs)

        print("\nClassification Report:")
        print(classification_report(y_test, val_pred))

        return self.model

    def _plot_metrics(self, y_true, y_pred, n_epochs):
        """Визуализация метрик обучения с несколькими эпохами"""
        plt.figure(figsize=(15, 5))
        sns.set_style("whitegrid")

        # График точности
        plt.subplot(1, 3, 1)
        plt.plot(range(1, n_epochs + 1), self.train_accuracy, 'bo-', label='Train Accuracy')
        plt.plot(range(1, n_epochs + 1), self.val_accuracy, 'ro-', label='Validation Accuracy')
        plt.title('Model Accuracy', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(range(1, n_epochs + 1))
        plt.ylim(0, 1.1)
        plt.legend()

        # График потерь
        plt.subplot(1, 3, 2)
        plt.plot(range(1, n_epochs + 1), self.train_loss, 'go-', label='Train Loss (1 - accuracy)')
        plt.title('Model Loss', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(1, n_epochs + 1))
        plt.legend()

        # Матрица ошибок
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix', fontsize=12)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.tight_layout()
        plt.savefig(self.metrics_path)
        plt.show()

    def _save_model(self):
        """Сохранение модели на диск"""
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Загрузка модели с диска"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please train the model first by calling .train()"
            )
        self.model = joblib.load(self.model_path)
        return self.model

    def predict(self, text: str) -> Tuple[str, float]:
        """Предсказание намерения с вероятностью"""
        if not hasattr(self, 'model'):
            self.load_model()

        clean_text = self._clean_text(text)
        if not clean_text:
            return "unknown", 0.0

        intent = self.model.predict([clean_text])[0]
        decision_scores = self.model.decision_function([clean_text])[0]
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probs = exp_scores / np.sum(exp_scores)
        confidence = np.max(probs)

        return intent, float(confidence)

    def _clean_text(self, text: str) -> str:
        """Очистка текста"""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def get_response(self, text: str, default_response: str = "Извините, я не понял вопрос") -> str:
        """Получить ответ бота на текст пользователя"""
        intent, confidence = self.predict(text)

        if confidence < 0.3:
            return default_response

        responses = self.config['intents'].get(intent, {}).get('responses', [])
        return np.random.choice(responses) if responses else default_response

# Пример использования
if __name__ == "__main__":
    # Инициализация и обучение
    classifier = IntentClassifier()
    classifier.train(n_epochs=15)

    # Тестирование
    test_phrases = [
        "Как купить билет на поезд?",
        "Где найти расписание?",
        "Какие есть скидки?",
        "Как вернуть билет?"
    ]

    for phrase in test_phrases:
        response = classifier.get_response(phrase)
        print(f"Вопрос: {phrase}")
        print(f"Ответ: {response}\n")

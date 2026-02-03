from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

class PriorityPredictor:
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self._train_dummy_model()

    def _train_dummy_model(self):
        # Initial dummy data to bootstrap the model
        texts = [
            "fix critical bug in production",
            "server is down",
            "update button color",
            "add new feature for user profile",
            "typo in documentation",
            "database connection failed",
            "security vulnerability found",
            "change font size",
            "refactor code",
            "implement login page"
        ]
        labels = [
            "CRITICAL",
            "CRITICAL",
            "LOW",
            "MEDIUM",
            "LOW",
            "HIGH",
            "CRITICAL",
            "LOW",
            "MEDIUM",
            "HIGH"
        ]
        self.model.fit(texts, labels)

    def predict(self, text: str) -> str:
        return self.model.predict([text])[0]

predictor = PriorityPredictor()

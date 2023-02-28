""" Sentiment Analysis of tweets.

https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
"""
from spacy.tokens import Doc
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from spacy import Language

from juxtorpus.corpus.processors.components import Component

import colorlog

logger = colorlog.getLogger(__name__)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"


class Sentiment(object):
    def __init__(self):
        self.MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        logger.info(f"Loading sentiment model: {self.MODEL}... Please wait.")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.config = AutoConfig.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        logger.info(f"Successfully loaded.")

    def score(self, text: str) -> dict[str, float]:
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        results = dict()
        for i in range(scores.shape[0]):
            l = self.config.id2label[ranking[i]]
            s = scores[ranking[i]]
            results[l] = np.round(float(s), 4)
        return results

    def preprocess(self, text):
        # Preprocess text (username and link placeholders)
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)


class SentimentComp(Component):
    def __init__(self, nlp: Language, name: str, attr: str):
        super(SentimentComp, self).__init__(nlp, name, attr)
        if Doc.has_extension(self._attr):
            raise KeyError(f"{self._attr} already exists. {SentimentComp.__name__} will not function properly.")
        Doc.set_extension(self._attr, default={})
        self.sentiment = None

    def __call__(self, doc: Doc) -> Doc:
        self.load_sentiment_model()
        text = doc.text
        scores = self.sentiment.score(text)
        doc._.set(self.attr, scores)
        return doc

    def load_sentiment_model(self):
        if self.sentiment is None:
            self.sentiment = Sentiment()


if __name__ == '__main__':
    text = "Covid cases are increasing fast!"

    sentiment = Sentiment()
    sentiments = sentiment.score(text)
    print(sentiments)

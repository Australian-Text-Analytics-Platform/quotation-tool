from abc import ABCMeta, abstractmethod

from juxtorpus.corpus import Corpus


class ProcessEpisode(object):
    def __init__(self, message: str):
        self.message = message

    def __repr__(self):
        return f"<{self.__class__.__name__}> {self.message}"


class Processor(metaclass=ABCMeta):
    def run(self, corpus: Corpus):
        corpus = self._process(corpus)
        self._add_metas(corpus)
        corpus.add_process_episode(self._create_episode())
        return corpus

    @abstractmethod
    def _process(self, corpus: Corpus):
        raise NotImplementedError()

    @abstractmethod
    def _add_metas(self, corpus: Corpus):
        raise NotImplementedError()

    def _create_episode(self) -> ProcessEpisode:
        return ProcessEpisode("No message was set.")


from .spacy_processor import SpacyProcessor


def process(corpus, **kwargs):
    if not isinstance(corpus, Corpus): raise ValueError(f"corpus is not an instance of {Corpus.__name__}.")
    if 'nlp' in kwargs.keys():
        return SpacyProcessor(kwargs.get('nlp')).run(corpus)
    raise ValueError(f"Only SpacyProcessor is supported at the moment. Please supply nlp argument.")

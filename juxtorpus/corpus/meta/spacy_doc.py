from .base import Meta
from spacy import Language
from spacy.tokens import Doc
from typing import Union, Callable, Iterable, Any
import pandas as pd
from functools import partial


class DocMeta(Meta):
    """ This class represents the metadata stored within the spacy Docs """

    def __init__(self, id_: str, attr: str,
                 nlp: Language, docs: Union[pd.Series, Callable[[], Iterable[Doc]]]):
        super(DocMeta, self).__init__(id_)
        self._attr = attr
        self._docs = docs
        self._nlp = nlp  # keep a ref to the spacy.Language

    @property
    def attr(self):
        return self._attr

    def apply(self, func) -> pd.Series:
        def _inner_func_on_attr(doc: Doc):
            return func(self._get_doc_attr(doc))

        if isinstance(self._docs, pd.Series):
            return self._docs.apply(_inner_func_on_attr)
        return pd.Series(map(_inner_func_on_attr, self._docs()))  # faster than loop. But can be improved.

    def cloned(self, texts, mask):
        # use the series mask to clone itself.
        if isinstance(self._docs, pd.Series):
            return DocMeta(self._id, self._attr, self._nlp, self._docs[mask])
        return DocMeta(self._id, self._attr, self._nlp, partial(self._nlp.pipe, texts))

    def head(self, n: int):
        docs = self._get_iterable()
        texts = (doc.text for i, doc in enumerate(docs) if i < n)
        attrs = (self._get_doc_attr(doc) for i, doc in enumerate(docs) if i < n)
        return pd.DataFrame(zip(texts, attrs), columns=['text', self._id])

    def __iter__(self):
        for doc in self._get_iterable():
            yield doc

    def _get_iterable(self):
        docs: Iterable
        if isinstance(self._docs, pd.Series):
            docs = self._docs
        elif isinstance(self._docs, Callable):
            docs = self._docs()
        else:
            raise RuntimeError(f"docs are neither a Series or a Callable stream. This should not happen.")
        return docs

    def _get_doc_attr(self, doc: Doc) -> Any:
        """ Returns a built-in spacy entity OR a custom entity. """
        # return doc.get_extension(self._attr) if doc.has_extension(self._attr) else getattr(doc, self._attr)
        return getattr(getattr(doc, '_'), self._attr) if doc.has_extension(self._attr) else getattr(doc, self._attr)

    def __repr__(self):
        return f"{super(DocMeta, self).__repr__()[:-2]}, Attribute: {self._attr}]"

    def summary(self) -> pd.DataFrame:
        # todo: return empty dataframe for now
        return pd.DataFrame(index=[self.id])

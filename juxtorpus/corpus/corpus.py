from typing import Dict, Generator, Optional, Callable
import pandas as pd
import spacy.vocab
from spacy.tokens import Doc
from sklearn.feature_extraction.text import CountVectorizer
import re

from juxtorpus.corpus.meta import MetaRegistry, Meta, SeriesMeta
from juxtorpus.corpus.dtm import DTM
from juxtorpus.matchers import is_word

import logging

logger = logging.getLogger(__name__)


class Corpus:
    """ Corpus
    This class abstractly represents a corpus which is a collection of documents.
    Each document is also described by their metadata and is used for functions such as slicing.

    An important component of the Corpus is that it also holds the document-term matrix which you can access through
    the accessor `.dtm`. See class DTM. The dtm is lazily loaded and is always computed for the root corpus.
    (read further for a more detailed explanation.)

    A main design feature of the corpus is to allow for easy slicing and dicing based on the associated metadata,
    text in document. See class CorpusSlicer. After each slicing operation, new but sliced Corpus object is
    returned exposing the same descriptive functions (e.g. summary()) you may wish to call again.

    To build a corpus, use the CorpusBuilder. This class handles the complexity

    ```
    builder = CorpusBuilder(pathlib.Path('./data.csv'))
    builder.add_metas('some_meta', 'datetime')
    builder.set_text_column('text')
    corpus = builder.build()
    ```

    Internally, documents are stored as rows of string in a dataframe. Metadata are stored in the meta registry.
    Slicing is equivalent to creating a `cloned()` corpus and is really passing a boolean mask to the dataframe and
    the associated metadata series. When sliced, corpus objects are created with a reference to its parent corpus.
    This is mainly for performance reasons, so that the expensive DTM computed may be reused and a shared vocabulary
    is kept for easier analysis of different sliced sub-corpus. You may choose the corpus to be `detached()` from this
    behaviour, and the corpus will act as the root, forget its lineage and a new dtm will need to be rebuilt.
    """

    COL_TEXT: str = 'text'

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, col_text: str = COL_TEXT):
        if col_text not in df.columns:
            raise ValueError(f"Column {col_text} not found. You must set the col_text argument.\n"
                             f"Available columns: {df.columns}")
        meta_df: pd.DataFrame = df.drop(col_text, axis=1)
        metas: dict[str, SeriesMeta] = dict()
        for col in meta_df.columns:
            # create series meta
            if metas.get(col, None) is not None:
                raise KeyError(f"{col} already exists. Please rename the column.")
            metas[col] = SeriesMeta(col, meta_df.loc[:, col])
        return Corpus(df[col_text], metas)

    def __init__(self, text: pd.Series,
                 metas: Dict[str, Meta] = None):
        text.name = self.COL_TEXT
        self._df: pd.DataFrame = pd.DataFrame(text, columns=[self.COL_TEXT])
        # ensure initiated object is well constructed.
        assert len(list(filter(lambda x: x == self.COL_TEXT, self._df.columns))) <= 1, \
            f"More than 1 {self.COL_TEXT} column in dataframe."

        self._parent: Optional[Corpus] = None

        # meta data
        self._meta_registry = MetaRegistry(metas)

        # document term matrix - DTM
        self._dtm: Optional[DTM] = DTM()

        # processing
        self._processing_history = list()

    @property
    def parent(self):
        return self._parent

    @property
    def is_root(self):
        return self._parent is None

    # slicing
    @property
    def slicer(self):
        from juxtorpus.corpus import CorpusSlicer
        return CorpusSlicer(self)

    # document term matrix
    @property
    def dtm(self):
        """ Document-Term Matrix. """
        if not self._dtm.is_built:
            root = self.find_root()
            root._dtm.initialise(root.texts())
            # self._dtm.build(root.texts())        # dtm tracks root and builds with root anyway
        return self._dtm

    def find_root(self):
        """ Find and return the root corpus. """
        if self.is_root: return self
        parent = self._parent
        while not parent.is_root:
            parent = parent._parent
        return parent

    def create_custom_dtm(self, tokeniser_func: Callable[[str], list[str]]):
        """ Create a custom DTM based on custom tokeniser function. """
        dtm = DTM()
        dtm.initialise(self.texts(),
                       vectorizer=CountVectorizer(preprocessor=lambda x: x, tokenizer=tokeniser_func))
        return dtm

    # meta data
    @property
    def meta(self):
        return self._meta_registry.copy()

    # processing
    def history(self):
        """ Returns a list of processing history. """
        return self._processing_history.copy()

    def add_process_episode(self, episode):
        self._processing_history.append(episode)

    # statistics
    @property
    def num_terms(self) -> int:
        return self.dtm.total

    @property
    def unique_terms(self) -> set[str]:
        return set(self.dtm.vocab(nonzero=True))

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT]

    def summary(self):
        """ Basic summary statistics of the corpus. """
        docs_info = pd.Series(self.dtm.total_docs_vector).describe().drop("count")
        # docs_info = docs_info.loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        mapper = {row_idx: f"No. Terms {row_idx}" for row_idx in docs_info.index}
        docs_info.rename(index=mapper, inplace=True)

        other_info = pd.Series({
            "Corpus Type": self.__class__.__name__,
            "No. Documents": len(self),
            "No. Terms": self.dtm.total,
            "Vocabulary size": len(self.dtm.vocab(nonzero=True)),
        })

        meta_info = pd.Series({
            "metas": ', '.join(self._meta_registry.keys())
        })
        return pd.concat([other_info, docs_info, meta_info])

    def generate_words(self):
        """ Generate list of words for each document in the corpus. """
        texts = self.texts()
        for i in range(len(texts)):
            yield self._gen_words_from(texts.iloc[i])

    def _gen_words_from(self, text) -> Generator[str, None, None]:
        return (token.lower() for token in re.findall('[A-Za-z]+', text))

    def generate_tokens(self):
        texts = self.texts()
        for i in range(len(texts)):
            yield self._gen_tokens_from(texts.iloc[i])

    def _gen_tokens_from(self, text) -> Generator[str, None, None]:
        return (token.lower() for token in text.split(" "))

    def cloned(self, mask: 'pd.Series[bool]'):
        """ Returns a (usually smaller) clone of itself with the boolean mask applied. """
        cloned_texts = self._cloned_texts(mask)
        cloned_metas = self._cloned_metas(mask)

        clone = Corpus(cloned_texts, cloned_metas)
        clone._parent = self

        clone._dtm = self._cloned_dtm(cloned_texts.index)
        clone._processing_history = self._cloned_history()
        return clone

    def _cloned_texts(self, mask):
        return self.texts()[mask]

    def _cloned_metas(self, mask):
        cloned_meta_registry = dict()
        for id_, meta in self._meta_registry.items():
            cloned_meta_registry[id_] = meta.cloned(texts=self._df.loc[:, self.COL_TEXT], mask=mask)
        return cloned_meta_registry

    def _cloned_history(self):
        return [h for h in self.history()]

    def _cloned_dtm(self, indices):
        return self._dtm.cloned(indices)

    def detached(self):
        """ Detaches from corpus tree and becomes the root.

        DTM will be regenerated when accessed - hence a different vocab.
        """
        self._parent = None
        self._dtm = DTM()
        meta_reg = MetaRegistry()
        for k, meta in self.meta.items():
            if isinstance(meta, SeriesMeta):
                sm = SeriesMeta(meta.id, meta.series().copy().reset_index(drop=True))
                meta_reg[sm.id] = sm
            else:
                meta_reg[k] = meta
        self._meta_registry = meta_reg
        self._df = self._df.copy().reset_index(drop=True)
        return self

    def __len__(self):
        return len(self._df) if self._df is not None else 0

    def __iter__(self):
        col_text_idx = self._df.columns.get_loc('text')
        for i in range(len(self)):
            yield self._df.iat[i, col_text_idx]

    def __getitem__(self, item):
        if isinstance(item, int):
            mask = self._df.index == self._df.iloc[item].name
        else:  # i.e. type=slice
            start = item.start
            stop = item.stop
            if start is None: start = 0
            if stop is None: stop = len(self._df)
            if item.step is not None: raise NotImplementedError("Slicing with step is currently not implemented.")
            mask = self._df.iloc[start:stop].index
        return self.cloned(mask)

    def sample(self, n: int, rand_stat=None):
        """ Uniformly sample from the corpus. """
        mask = self._df.isna().squeeze()  # Return a mask of all False
        mask[mask.sample(n=n, random_state=rand_stat).index] = True
        return self.cloned(mask)


class SpacyCorpus(Corpus):
    """ SpacyCorpus
    This class inherits from the Corpus class with the added and adjusted functions to handle spacy's Doc data
    structure as opposed to string. However, the original string data structure is kept. These may be accessed via
    `.docs()` and `.texts()` respectively.

    Metadata in this class also includes metadata stored in Doc objects. See class meta/DocMeta. Which may again
    be used for slicing the corpus.

    To build a SpacyCorpus, you'll need to `process()` a Corpus object. See class SpacyProcessor. This will run
    the spacy process and update the corpus's meta registry. You'll still need to load spacy's Language object
    which is used in the process.

    ```
    nlp = spacy.blank('en')
    from juxtorpus.corpus.processors import process
    spcycorpus = process(corpus, nlp=nlp)
    ```

    Subtle differences to Corpus:
    As spacy utilises the tokeniser set out by the Language object, you may find summary statistics to be inconsistent
    with the Corpus object you had before it was processed into a SpacyCorpus.
    """

    @classmethod
    def from_corpus(cls, corpus: Corpus, docs, nlp):
        return cls(docs, corpus._meta_registry, nlp)

    def __init__(self, docs, metas, nlp: spacy.Language):
        super(SpacyCorpus, self).__init__(docs, metas)
        self._nlp = nlp
        self._is_word_matcher = is_word(self._nlp.vocab)
        # self._df.reset_index(inplace=True)

    @property
    def nlp(self):
        return self._nlp

    @property
    def slicer(self):
        from juxtorpus.corpus import SpacyCorpusSlicer
        return SpacyCorpusSlicer(self)

    @property
    def dtm(self):
        if not self._dtm.is_built:
            root = self.find_root()
            root._dtm.initialise(root.docs(),
                                 vectorizer=CountVectorizer(preprocessor=lambda x: x,
                                                            tokenizer=self._gen_words_from))
        return self._dtm

    def create_custom_dtm(self, tokeniser_func: Callable[[str], list[str]]):
        """ Create a custom DTM with tokens returned by the tokeniser_func."""
        dtm = DTM()
        dtm.initialise(self.docs(), vectorizer=CountVectorizer(preprocessor=lambda x: x, tokenizer=tokeniser_func))
        return dtm

    def texts(self) -> 'pd.Series[str]':
        return self._df.loc[:, self.COL_TEXT].map(lambda doc: doc.text)

    def docs(self) -> 'pd.Series[Doc]':
        return self._df.loc[:, self.COL_TEXT]

    def _gen_words_from(self, doc):
        return (doc[start: end].text.lower() for _, start, end in self._is_word_matcher(doc))

    def generate_lemmas(self):
        texts = self.texts()
        for i in range(len(texts)):
            yield self._gen_lemmas_from(texts.iloc[i])

    def _gen_lemmas_from(self, doc):
        return (doc[start: end].lemma_ for _, start, end in self._is_word_matcher(doc))

    def _cloned_docs(self, mask):
        return self.docs().loc[mask]

    def cloned(self, mask: 'pd.Series[bool]'):
        # cloned_texts = self._cloned_texts(mask)
        cloned_docs = self._cloned_docs(mask)
        cloned_metas = self._cloned_metas(mask)

        clone = SpacyCorpus(cloned_docs, cloned_metas, self._nlp)
        clone._parent = self

        clone._dtm = self._cloned_dtm(cloned_docs.index)
        clone._processing_history = self._cloned_history()
        return clone

    def summary(self, spacy: bool = False):
        df = super(SpacyCorpus, self).summary()
        if spacy:
            spacy_info = {
                'lang': self.nlp.meta.get('lang'),
                'model': self.nlp.meta.get('name'),
                'pipeline': ', '.join(self.nlp.pipe_names)
            }
            return pd.concat([df, pd.DataFrame.from_dict(spacy_info, orient='index')])
        return df

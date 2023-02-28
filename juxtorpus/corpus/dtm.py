import contextlib
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Iterable, TypeVar
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import scipy.sparse
import logging

logger = logging.getLogger(__name__)

from juxtorpus.corpus.freqtable import FreqTable

""" Document Term Matrix DTM

DTM is a container for the document term sparse matrix.
This container allows you to access term vectors and document vectors.
It also allows you to clone itself with a row index. In the cloned DTM,
a reference to the root dtm is passed down and the row index is used to
slice the child dtm each time.
This serves 3 purposes:
    1. preserve the original/root vocabulary.
    2. performance reasons so that we don't need to rebuild a dtm each time.
    3. indexing is a very inexpensive operation.

Dependencies: 
sklearn CountVectorizer
"""

TVectorizer = TypeVar('TVectorizer', bound=CountVectorizer)


class DTM(object):
    """ DTM
    This class is an abstract representation of the document-term matrix. It serves as a component
    of the Corpus class and exposes various functionalities that allows the slicing and dicing to be
    done seamlessly.

    Internally, DTM stores a sparse matrix which is computed using sklearn's CountVectorizer.
    """

    @classmethod
    def from_wordlists(cls, wordlists: Iterable[Iterable[str]]):
        return cls().initialise(wordlists)

    def __init__(self):
        self.root = self
        self._vectorizer = None
        self._matrix = None
        self._feature_names_out = None
        self._term_idx_map = None
        self._is_built = False
        self.derived_from = None  # for any dtms derived from word frequencies

        # only used for child dtms
        self._row_indices = None
        self._col_indices = None

    @property
    def is_built(self) -> bool:
        return self.root._is_built

    @property
    def matrix(self):
        matrix = self.root._matrix
        if self._row_indices is not None:
            matrix = matrix[self._row_indices, :]
        if self._col_indices is not None:
            matrix = matrix[:, self._col_indices]
        return matrix

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def num_terms(self):
        return self.matrix.shape[1]

    @property
    def num_docs(self):
        return self.matrix.shape[0]

    @property
    def total(self):
        return self.matrix.sum()

    @property
    def total_terms_vector(self):
        """ Returns a vector of term counts for each term. """
        return np.asarray(self.matrix.sum(axis=0)).squeeze(axis=0)

    @property
    def total_docs_vector(self):
        """ Returns a vector of term counts for each document. """
        return np.asarray(self.matrix.sum(axis=1)).squeeze(axis=1)

    @property
    def vectorizer(self):
        return self.root._vectorizer

    @property
    def term_names(self):
        """ Return the terms in the current dtm. """
        features = self.root._feature_names_out
        return features if self._col_indices is None else features[self._col_indices]

    def vocab(self, nonzero: bool = False):
        """ Returns a set of terms in the current dtm. """
        if nonzero:
            return set(self.term_names[self.total_terms_vector.nonzero()[0]])
        else:
            return set(self.term_names)

    def initialise(self, texts: Iterable[str], vectorizer: TVectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')):
        logger.debug("Building document-term matrix. Please wait...")
        self.root._vectorizer = vectorizer
        self.root._matrix = self.root._vectorizer.fit_transform(texts)
        self.root._feature_names_out = self.root._vectorizer.get_feature_names_out()  # expensive operation - cached.
        self.root._term_idx_map = {self.root._feature_names_out[idx]: idx
                                   for idx in range(len(self.root._feature_names_out))}
        self.root._is_built = True
        logger.debug("Done.")
        return self

    def terms_column_vectors(self, terms: Union[str, list[str]]):
        """ Return the term vector represented by the documents. """
        cols: Union[int, list[int]]
        if isinstance(terms, str):
            cols = self._term_to_idx(terms)
        else:
            cols = [self._term_to_idx(term) for term in terms]
        return self.matrix[:, cols]

    def doc_vector(self):  # TODO: from pandas index?
        """ Return the document vector represented by the terms. """
        raise NotImplementedError()

    def _term_to_idx(self, term: str):
        if term not in self.root._term_idx_map.keys(): raise ValueError(f"'{term}' not found in document-term-matrix.")
        return self.root._term_idx_map.get(term)

    def cloned(self, row_indices: Union[pd.core.indexes.numeric.Int64Index, list[int]]):
        cloned = DTM()
        cloned.root = self.root
        cloned._row_indices = row_indices
        if cloned.is_built:
            try:
                cloned.matrix
            except Exception as e:
                raise RuntimeError([RuntimeError("Failed to clone DTM."), e])
        return cloned

    def tfidf(self, **kwargs):
        """ Returns an un-normalised tfidf of the current matrix. A new DTM is returned.

        Args: see sklearn.TfidfTransformer
        norm is set to None by default here.
        """
        kwargs['use_idf'] = kwargs.get('use_idf', True)
        kwargs['smooth_idf'] = kwargs.get('smooth_idf', True)
        kwargs['sublinear_tf'] = kwargs.get('sublinear_tf', False)
        kwargs['norm'] = kwargs.get('norm', None)
        tfidf_trans = TfidfTransformer(**kwargs)
        tfidf = DTM()
        tfidf.derived_from = self
        tfidf._vectorizer = tfidf_trans
        tfidf._matrix = tfidf._vectorizer.fit_transform(self.matrix)
        tfidf._feature_names_out = self.term_names
        tfidf._term_idx_map = {tfidf._feature_names_out[idx]: idx for idx in range(len(tfidf._feature_names_out))}
        tfidf._is_built = True
        return tfidf

    def to_dataframe(self):
        return pd.DataFrame.sparse.from_spmatrix(self.matrix, columns=self.term_names)

    @contextlib.contextmanager
    def without_terms(self, terms: Union[list[str], set[str]]):
        """ Expose a temporary dtm object without a list of terms. Terms not found are ignored. """
        try:
            features = self.root._feature_names_out
            self._col_indices = np.isin(features, set(terms), invert=True).nonzero()[0]
            yield self
        finally:
            self._col_indices = None

    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, np.matrix], terms):
        num_terms: int
        if isinstance(terms, list): num_terms = len(terms)
        elif isinstance(terms, np.ndarray) and len(terms.shape) == 1: num_terms = terms.shape[0]
        elif isinstance(terms, np.ndarray) and len(terms.shape) == 2: num_terms = terms.shape[1]
        else: raise ValueError(f"Expecting terms to be either list or array but got {type(terms)}.")
        assert matrix.shape[1] == num_terms, f"Mismatched terms. Matrix shape {matrix.shape} and {num_terms} terms."
        if not scipy.sparse.issparse(matrix):
            logger.warning(f"Accepted a non sparse matrix as DTM, this may be expensive in memory.")
        dtm = cls()
        dtm._matrix = matrix
        dtm._feature_names_out = terms
        dtm._is_built = True
        return dtm

    def shares_vocab(self, other: 'DTM') -> bool:
        """ Check if the other DTM shares current DTM's vocab """
        this, other = self.vocab(nonzero=True), other.vocab(nonzero=True)
        if not len(this) == len(other): return False
        return len(this.difference(other)) == 0

    def terms_aligned(self, other: 'DTM') -> bool:
        """ Check if the other DTM's terms are index aligned with current DTM """
        this, other = self.term_names, other.term_names
        if not len(this) == len(other): return False
        return (this == other).all()

    def merged(self, other: 'DTM'):
        """Merge other DTM with current."""
        if self.terms_aligned(other):
            m = scipy.sparse.vstack((self.matrix, other.matrix))
            feature_names_out = self._feature_names_out  # unchanged since terms are shared
        else:
            if len(other.term_names) >= len(self.term_names):
                big, small = other, self
            else:
                big, small = self, other

            top_right, indx_missing = self._build_top_right_merged_matrix(big, small)
            top_left = self._build_top_left_merged_matrix(small)
            top = scipy.sparse.hstack((top_left, top_right))

            num_terms_sm_and_bg = small.num_terms + indx_missing.shape[0]
            assert top.shape[0] == small.num_docs and top.shape[1] == num_terms_sm_and_bg, \
                f"Top matrix incorrect shape: Expecting ({small.num_docs, num_terms_sm_and_bg}. Got {top.shape}."

            bottom_left = self._build_bottom_left_merged_matrix(big, small)  # shape=(big.num_docs, small.num_terms)
            bottom_right = self._build_bottom_right_merged_matrix(big,
                                                                  indx_missing)  # shape=(big.num_docs, missing terms from big)
            bottom = scipy.sparse.hstack((bottom_left, bottom_right))
            logger.debug(f"MERGE: merged bottom matrix shape: {bottom.shape} type: {type(bottom)}.")
            assert bottom.shape[0] == big.num_docs and bottom_left.shape[1] == small.num_terms, \
                f"Bottom matrix incorrect shape: Expecting ({big.num_docs}, {num_terms_sm_and_bg}). Got {bottom.shape}."

            m = scipy.sparse.vstack((top, bottom))
            logger.debug(f"MERGE: merged matrix shape: {m.shape} type: {type(m)}.")
            assert m.shape[1] == num_terms_sm_and_bg, \
                f"Terms incorrectly merged. Total unique terms: {num_terms_sm_and_bg}. Got {m.shape[1]}."
            num_docs_sm_and_bg = big.num_docs + small.num_docs
            assert m.shape[0] == num_docs_sm_and_bg, \
                f"Documents incorrectly merged. Total documents: {num_docs_sm_and_bg}. Got {m.shape[0]}."
            feature_names_out = np.concatenate([small.term_names, big.term_names[indx_missing]])

        # replace with new matrix.
        other = DTM()
        other._matrix = m
        other._feature_names_out = feature_names_out
        return other

    def _build_top_right_merged_matrix(self, big, small):
        # 1. build top matrix: shape = (small.num_docs, small.num_terms + missing terms from big)
        # perf: assume_uniq - improves performance and terms are unique.
        mask_missing = np.isin(big.term_names, small.term_names, assume_unique=True, invert=True)
        indx_missing = mask_missing.nonzero()[0]
        # create zero matrix in top right since small doesn't have these terms in their documents.
        top_right = scipy.sparse.csr_matrix((small.num_docs, indx_missing.shape[0]), dtype=small.matrix.dtype)
        return top_right, indx_missing

    def _build_top_left_merged_matrix(self, small):
        return small.matrix

    def _build_bottom_left_merged_matrix(self, big, small):
        # 2. build bottom matrix: shape = (big.num_docs, small.num_terms + missing terms from big)
        # bottom-left: shape = (big.num_docs, small.num_terms)
        #   align overlapping term indices from big with small term indices
        intersect = np.intersect1d(big.term_names, small.term_names, assume_unique=True, return_indices=True)
        intersect_terms, bg_intersect_indx, sm_intersect_indx = intersect
        bottom_left = scipy.sparse.lil_matrix((big.num_docs, small.num_terms))  # perf: lil for column replacement
        for i, idx in enumerate(sm_intersect_indx):
            bottom_left[:, idx] = big.matrix[:, bg_intersect_indx[i]]
        logger.debug(f"MERGE: bottom left matrix shape: {bottom_left.shape}")
        bottom_left = bottom_left.tocsr(copy=False)  # convert to csr to match with rest of submatrices
        return bottom_left

    def _build_bottom_right_merged_matrix(self, big, indx_missing):
        return big.matrix[:, indx_missing]

    def freq_table(self, nonzero=True) -> FreqTable:
        """ Create a frequency table of the dataframe."""
        terms, freqs = self.term_names, self.total_terms_vector
        if nonzero:
            indices = np.nonzero(self.total_terms_vector)[0]
            terms, freqs = self.term_names[indices], self.total_terms_vector[indices]
        return FreqTable(terms, freqs)

    def __repr__(self):
        return f"<DTM {self.num_docs} docs X {self.num_terms} terms>"


if __name__ == '__main__':
    from juxtorpus.corpus.corpus import Corpus

    df = pd.read_csv(Path("./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv"))
    corpus = Corpus.from_dataframe(df, col_text='processed_text')
    corpus.summary()

    dtm = DTM().initialise(corpus.texts())
    print(dtm.terms_column_vectors('the').shape)
    print(dtm.terms_column_vectors(['the', 'he', 'she']).shape)

    sub_df = df[df['processed_text'].str.contains('the')]

    child_dtm = dtm.cloned(dtm, sub_df.index)
    print(child_dtm.terms_column_vectors('the').shape)
    print(child_dtm.terms_column_vectors(['the', 'he', 'she']).shape)

    df = child_dtm.to_dataframe()
    print(df.head())

    print(f"Child DTM shape: {child_dtm.shape}")
    print(f"Child DTM DF shape: {df.shape}")
    print(f"Child DTM DF memory usage:")
    df.info(memory_usage='deep')

    # with remove_words context
    prev = set(dtm.term_names)
    with dtm.without_terms({'hello'}) as subdtm:
        print(subdtm.num_terms)
        print(prev.difference(set(subdtm.term_names)))

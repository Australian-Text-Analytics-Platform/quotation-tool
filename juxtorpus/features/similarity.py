"""
Similarity between 2 Corpus.

1. jaccard similarity
2. pca similarity
"""
from scipy.spatial import distance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
from typing import TYPE_CHECKING, Union
import weakref as wr

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.freqtable import FreqTable
from juxtorpus.constants import CORPUS_ID_COL_NAME_FORMAT

if TYPE_CHECKING:
    from juxtorpus import Jux


def _cos_sim(v0: Union[np.ndarray, pd.Series], v1: Union[np.ndarray, pd.Series]):
    if isinstance(v0, np.ndarray) and isinstance(v1, np.ndarray):
        assert v0.ndim == 1 and v1.ndim == 1, "Must be 1d array."
        assert v0.shape[0] == v1.shape[0], f"Mismatched shape {v0.shape=} {v1.shape=}"
        if v0.shape[0] == 0: return 0
    elif isinstance(v0, pd.Series) and isinstance(v1, pd.Series):
        assert len(v0) == len(v1), f"Mismatched shape {len(v0)=} {len(v1)=}"
        if len(v0) == 0: return 0
    else:
        raise ValueError(f"They must both be either "
                         f"{np.ndarray.__class__.__name__} or "
                         f"{pd.Series.__class__.__name__}.")
    return 1 - distance.cosine(v0, v1)


class Similarity(object):
    def __init__(self, jux: 'Jux'):
        self._jux = wr.ref(jux)

    def jaccard(self, use_lemmas: bool = False):
        """ Return a similarity score between the 2 corpus."""
        if use_lemmas:
            # check if corpus are spacy corpus.
            raise NotImplementedError("To be implemented. Use unique lemmas instead of words.")
        u0: set[str] = self._jux().corpus_0.unique_terms
        u1: set[str] = self._jux().corpus_1.unique_terms
        return len(u0.intersection(u1)) / len(u0.union(u1))

    def lsa_pairwise_cosine(self, n_components: int = 100, verbose=False):
        """ Decompose DTM to SVD and return the pairwise cosine similarity of the right singular matrix.

        Note: this may be different to the typical configuration using a TDM instead of DTM.
        However, sklearn only exposes the right singular matrix.
        tdm.T = (U Sigma V.T).T = V.T.T Sigma.T U.T = V Sigma U.T
        the term-topic matrix of U is now the right singular matrix if we use DTM instead of TDM.
        """
        A, B = self._jux().corpus_0, self._jux().corpus_1
        svd_A = TruncatedSVD(n_components=n_components).fit(A.dtm.tfidf().matrix)
        svd_B = TruncatedSVD(n_components=n_components).fit(B.dtm.tfidf().matrix)
        top_topics = 5
        if verbose:
            top_terms = 5
            for corpus, svd in [(A, svd_A), (B, svd_B)]:
                feature_indices = svd.components_.argsort()[::-1][
                                  :top_topics]  # highest value term in term-topic matrix
                terms = corpus.dtm.term_names[feature_indices]
                for i in range(feature_indices.shape[0]):
                    print(f"Corpus {str(corpus)}: Singular columns [{i}] {terms[i][:top_terms]}")

        # pairwise cosine
        return cosine_similarity(svd_A.components_[:top_topics], svd_B.components_[:top_topics])

    def cosine_similarity(self, metric: str, *args, **kwargs):
        metric_map = {
            'tf': self._cos_sim_tf,
            'tfidf': self._cos_sim_tfidf,
            'log_likelihood': self._cos_sim_llv
        }
        sim_fn = metric_map.get(metric, None)
        if sim_fn is None: raise ValueError(f"Only metrics {metric_map.keys()} are supported.")
        return sim_fn(*args, **kwargs)

    def _cos_sim_llv(self, baseline: FreqTable = None):
        if baseline is None:
            corpora = self._jux().corpora
            baseline = FreqTable.from_freq_tables([corpus.dtm.freq_table(nonzero=True) for corpus in corpora])

        res = self._jux().stats.log_likelihood_and_effect_size(baseline=baseline).fillna(0)
        return _cos_sim(res[CORPUS_ID_COL_NAME_FORMAT.format('log_likelihood_llv', 0)],
                        res[CORPUS_ID_COL_NAME_FORMAT.format('log_likelihood_llv', 1)])

    def _cos_sim_tf(self, without: list[str] = None) -> float:
        seriess = list()
        for i, corpus in enumerate(self._jux().corpora):
            ft = corpus.dtm.freq_table(nonzero=True)
            if without: ft.remove(without)
            seriess.append(ft.series.rename(CORPUS_ID_COL_NAME_FORMAT.format(ft.name, i)))

        res = pd.concat(seriess, axis=1).fillna(0)
        return _cos_sim(res.iloc[:, 0], res.iloc[:, 1])

    def _cos_sim_tfidf(self, **kwargs):
        seriess = list()
        for i, corpus in enumerate(self._jux().corpora):
            ft = corpus.dtm.tfidf(**kwargs).freq_table(nonzero=True)
            seriess.append(ft.series.rename(CORPUS_ID_COL_NAME_FORMAT.format(ft.name, i)))
        res = pd.concat(seriess, axis=1).fillna(0)
        return _cos_sim(res.iloc[:, 0], res.iloc[:, 1])


if __name__ == '__main__':
    import pandas as pd
    from juxtorpus.corpus import CorpusSlicer
    from juxtorpus.jux import Jux

    corpus = Corpus.from_dataframe(
        pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                    usecols=['processed_text', 'tweet_lga']),
        # pd.read_csv('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv',
        #             usecols=['processed_text', 'tweet_lga']),
        col_text='processed_text'
    )

    slicer = CorpusSlicer(corpus)
    brisbane = slicer.filter_by_item('tweet_lga', 'Brisbane (C)')
    fairfield = slicer.filter_by_item('tweet_lga', 'Fairfield (C)')

    sim = Jux(brisbane, fairfield).sim
    pairwise = sim.lsa_pairwise_cosine(n_components=100, verbose=True)
    print(f"SVD pairwise cosine of PCs\n{pairwise}")
    jaccard = sim.jaccard()
    print(f"{jaccard=}")

    sw = nltk.corpus.stopwords.words('english')
    term_vec_cos = sim.cosine_similarity(metric='tf', without=sw + ['climate', 'tweet', 'https'])
    print(f"{term_vec_cos=}")

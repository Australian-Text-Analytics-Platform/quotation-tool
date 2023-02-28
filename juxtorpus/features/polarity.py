""" Polarity

This module handles the calculations of polarity of terms based on a chosen metric.

Metrics:
1. term frequency (normalised on total terms)
2. tfidf
3. log likelihood

Output:
-> dataframe
"""

from typing import TYPE_CHECKING, Optional
import weakref as wr
import pandas as pd
from typing import Generator
from juxtorpus.corpus.dtm import DTM

if TYPE_CHECKING:
    from juxtorpus import Jux


class Polarity(object):
    """ Polarity
    Gives a 'polarity' score of the two corpus based on token statistics.
    A polarity score uses 0 as the midline, a positive score means corpus 0 is dominant. And vice versa.
    e.g. tf would use the term frequencies to use as the polarity score.

    Each function will return a dataframe with a 'polarity' column and other columns with values that composes
    the 'polarity' score.
    """

    def __init__(self, jux: 'Jux'):
        self._jux: wr.ref['Jux'] = wr.ref(jux)

    def tf(self, tokeniser_func: Optional = None):
        """ Uses the term frequency to produce the polarity score.

        Polarity = Corpus 0's tf - Corpus 1's tf.
        """
        dtms = self._selected_dtms(tokeniser_func)
        fts = (dtm.freq_table() for dtm in dtms)

        renamed_ft = [(f"{ft.name}_corpus_{i}", ft) for i, ft in enumerate(fts)]
        df = pd.concat([ft.series.rename(name) / ft.total for name, ft in renamed_ft], axis=1).fillna(0)
        df['polarity'] = df[renamed_ft[0][0]] - df[renamed_ft[1][0]]
        return df

    def tfidf(self, tokeniser_func: Optional = None):
        """ Uses the tfidf scores to produce the polarity score.

        Polarity = Corpus 0's tfidf - Corpus 1's tfidf.
        """
        dtms = self._selected_dtms(tokeniser_func)
        fts = (dtm.freq_table() for dtm in dtms)
        renamed_ft = [(f"{ft.name}_corpus_{i}", ft) for i, ft in enumerate(fts)]
        df = pd.concat([ft.series.rename(name) / ft.total for name, ft in renamed_ft], axis=1).fillna(0)
        df['polarity'] = df[renamed_ft[0][0]] - df[renamed_ft[1][0]]
        return df

    def log_likelihood(self, tokeniser_func: Optional = None):
        j = self._jux()
        llv = j.stats.log_likelihood_and_effect_size()
        tf_polarity = self.tf(tokeniser_func)['polarity']
        llv['polarity'] = (tf_polarity * llv['log_likelihood_llv']) / tf_polarity.abs()
        return llv

    def _selected_dtms(self, tokeniser_func: Optional) -> Generator[DTM, None, None]:
        """ Return a generator DTMs given a tokeniser function."""
        if tokeniser_func:
            return (corpus.create_custom_dtm(tokeniser_func) for corpus in self._jux().corpora)
        else:
            return (corpus.dtm for corpus in self._jux().corpora)

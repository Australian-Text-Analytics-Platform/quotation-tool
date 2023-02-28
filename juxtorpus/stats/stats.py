from typing import TYPE_CHECKING

import pandas as pd
from .loglikelihood_effectsize import log_likelihood_and_effect_size
from juxtorpus.corpus.freqtable import FreqTable
from juxtorpus.constants import CORPUS_ID_COL_NAME_FORMAT

if TYPE_CHECKING:
    from juxtorpus import Jux


class Statistics(object):
    def __init__(self, jux: 'Jux'):
        self._jux = jux

    def log_likelihood_and_effect_size(self, baseline: FreqTable = None):
        ftables = [self._jux.corpus_0.dtm.freq_table(nonzero=True),
                   self._jux.corpus_1.dtm.freq_table(nonzero=True)]

        if baseline is None:
            res = log_likelihood_and_effect_size(ftables)
            res = res.filter(regex=r'(log_likelihood_llv|bayes_factor_bic|effect_size_ell)')
        else:
            res_list = list()
            for i, ft in enumerate(ftables):
                res = log_likelihood_and_effect_size([ft, baseline])
                res = res.filter(regex=r'(log_likelihood_llv|bayes_factor_bic|effect_size_ell)')
                mapper = dict()
                for col in res.columns:
                    mapper[col] = CORPUS_ID_COL_NAME_FORMAT.format(col, i)
                res.rename(mapper=mapper, axis=1, inplace=True)
                res_list.append(res)
            res = pd.concat(res_list, axis=1)
        # reformat to be consistent with jux
        return res

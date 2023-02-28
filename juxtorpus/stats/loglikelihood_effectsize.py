""" Log likelihood Ratios

Implementations of log likelihood ratios and effect size.

source: https://ucrel.lancs.ac.uk/llwizard.html
"""
import numpy as np
import pandas as pd

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.freqtable import FreqTable


def log_likelihood_and_effect_size(freq_tables: list[FreqTable]):
    """ Calculate the sum of log likelihood ratios over the corpora. """
    # res = pd.concat((corpus.dtm.freq_table(nonzero=True).df for corpus in corpora), axis=1)
    res = pd.concat((ft.series.rename(f"{ft.name}_corpus_{i}") for i, ft in enumerate(freq_tables)), axis=1)

    corpora_freqs = res.sum(axis=1)
    corpora_freqs_total = corpora_freqs.sum(axis=0)
    res['expected_likelihoods'] = corpora_freqs / corpora_freqs_total

    for i, ftable in enumerate(freq_tables):
        observed = res[f"{ftable.name}_corpus_{i}"]
        res[f"expected_freq_corpus_{i}"] = res['expected_likelihoods'] * observed.sum(axis=0)
        res[f"log_likelihood_ratio_corpus_{i}"] = _loglikelihood_ratios(res[f"expected_freq_corpus_{i}"], observed)
        res[f"log_likelihood_corpus_{i}"] = _weighted_log_likehood_ratios(res[f"expected_freq_corpus_{i}"], observed)

    res['log_likelihood_llv'] = res.filter(regex=r'log_likelihood_corpus_[0-9]+').sum(axis=1)
    res['bayes_factor_bic'] = _bayes_factor_bic(len(freq_tables), corpora_freqs_total, res['log_likelihood_llv'])
    min_expected = res.filter(regex=r'expected_freq_corpus_[0-9]+').min(axis=1)
    res['effect_size_ell'] = _effect_size_ell(min_expected, corpora_freqs_total, res['log_likelihood_llv'])
    return res


def _bayes_factor_bic(corpora_size: int, corpora_freq_total: int, log_likelihood: pd.Series):
    # if corpora_freq_total == 0: return 0
    dof = corpora_size - 1
    return log_likelihood - (dof * np.log(corpora_freq_total))


def _effect_size_ell(min_expected, corpora_freq_total: int, log_likelihood: pd.Series):
    # if min_expected == 0: return 0
    return log_likelihood / (corpora_freq_total * np.log(min_expected))


def _weighted_log_likehood_ratios(expected: pd.Series, observed: pd.Series):
    return 2 * np.multiply(observed, _loglikelihood_ratios(expected, observed))


def _loglikelihood_ratios(expected: pd.Series, observed: pd.Series):
    return np.log(observed) - np.log(expected)

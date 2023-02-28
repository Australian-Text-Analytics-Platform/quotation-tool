import pandas as pd
from typing import Union, Optional
from collections import Counter
import numpy as np


class FreqTable(object):
    """ Frequency Table
    The frequency table is an abstraction on top of the DTM.
    It is motivated by the
    """

    @classmethod
    def from_counter(cls, counter: Counter):
        return cls(terms=counter.keys(), freqs=counter.values())

    @classmethod
    def from_freq_tables(cls, freq_tables: list['FreqTable']):
        merged = FreqTable(list(), list())
        for ft in freq_tables: merged.merge(ft)
        return merged

    def __init__(self, terms, freqs):
        if len(terms) != len(freqs): raise ValueError(f"Mismatched terms and freqs. {terms=} {freqs=}.")
        if len(set(terms)) != len(terms): raise ValueError(f"Terms must be unique.")

        self._COL_FREQ = 'freq'
        self._series: pd.Series = pd.Series(freqs, index=terms, dtype=np.int, name=self._COL_FREQ)

    @property
    def series(self):
        return self._series

    @property
    def terms(self):
        return self._series.index.tolist()

    @property
    def freqs(self):
        return self._series.tolist()

    @property
    def total(self):
        return int(self._series.sum(axis=0))

    @property
    def name(self):
        return self._series.name

    def merge(self, other: Union['FreqTable', list[str]], freqs: Optional[list[int]] = None):
        """ Merge with another FreqTable. Or term, freq pair)"""
        if freqs is not None:  # overloaded method - allows term, freqs as well
            if isinstance(other, FreqTable):
                raise ValueError(f"You must use term freq pairs. Not {self.__class__.__name__}.")
            other = FreqTable(other, freqs)
        self._series: pd.Series = pd.concat([self._series, other.series], axis=1).fillna(0).sum(axis=1)

    def remove(self, terms: Union[str, list[str]]):
        """ Remove terms from frequency table. Ignored if not exist."""
        terms = list(terms)
        self.series.drop(terms, errors='ignore', inplace=True)

    def __getitem__(self, item: Union[str, int]):
        type_ = type(item)
        if not isinstance(item, (str, int)): raise ValueError("Must be either an integer or string")
        if type_ == str:
            # todo: if not exist return 0
            return self._series.loc[item]
        if type_ == int:
            # todo: if not exists, return empty list
            return self._series[self._series == item]

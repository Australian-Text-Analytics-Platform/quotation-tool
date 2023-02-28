from .base import Meta
from typing import Union
import pandas as pd

from juxtorpus.loader import LazySeries


class SeriesMeta(Meta):
    dtypes = {'float', 'float16', 'float32', 'float64',
              'int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
              'str', 'bool', 'category'}

    def __init__(self, id_, series: Union[pd.Series, LazySeries]):
        super(SeriesMeta, self).__init__(id_)
        self._series = series
        # print(self._series)

    def series(self):
        if isinstance(self._series, LazySeries):
            return self._series.load()
        return self._series

    def apply(self, func):
        return self.series().apply(func)

    def __iter__(self):
        for x in iter(self.series().__iter__()):
            yield x

    def cloned(self, texts, mask):
        return SeriesMeta(self._id, self.series().loc[mask])

    def head(self, n):
        return self.series().head(n)

    def summary(self) -> pd.DataFrame:
        """ Return a summary of the series in a dataframe. """
        series = self.series()

        # dtype
        info = {'dtype': series.dtype,
                'sample': series.iloc[0]}

        # uniques
        if self._show_uniqs(series):
            vc = series.value_counts(ascending=False).head(1)
            info['top'] = str(vc.index.values[0])
            info['top_freq'] = vc.values[0]

            uniqs = series.unique()
            info['uniqs'] = [uniqs]
            info['num_uniqs'] = len(uniqs)

        # mean, min, max, quantiles
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            info['mean'] = series.mean()
            info['std'] = series.std()
            info['min'] = series.min()
            info['25%'] = series.quantile(0.25)
            info['50%'] = series.quantile(0.5)
            info['75%'] = series.quantile(0.75)
            info['max'] = series.max()

        df = pd.DataFrame(info, index=[self.id])
        return df

    def _show_uniqs(self, series) -> bool:
        uniqs = series.unique()
        if pd.api.types.is_datetime64_any_dtype(series): return False
        if series.dtype == 'category': return True
        if len(uniqs) < 12: return True  # hard cap
        return False


"""Example Child Class (Archived): 

class DelimitedStrSeriesMeta(SeriesMeta):
    def __init__(self, id_, series: pd.Series, delimiter: str):
        super(DelimitedStrSeriesMeta, self).__init__(id_, series)
        self.delimiter = delimiter

    def apply(self, func):
        return self.series().apply(lambda x: x.split(self.delimiter)).apply(func)

    def cloned(self, texts, mask):
        return DelimitedStrSeriesMeta(self._id, self.series()[mask], self.delimiter)

"""

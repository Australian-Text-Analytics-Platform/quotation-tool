import weakref
from abc import ABCMeta, abstractmethod
from atap_widgets.concordance import ConcordanceTable, ConcordanceWidget
from juxtorpus.corpus import Corpus
import pandas as pd
from typing import Union


class Concordance(metaclass=ABCMeta):
    """ Concordance
    This is the base concordance class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def set_keyword(self, keyword: str):
        return self

    @abstractmethod
    def find(self):
        raise NotImplementedError()


class ATAPConcordance(ConcordanceWidget, Concordance):
    """ ATAPConcordance

    This implementation integrates with the Concordance tool from atap_widgets package.
    """

    def __init__(self, corpus: Corpus):
        super(ATAPConcordance, self).__init__(df=corpus.texts().to_frame('spacy_doc'))
        # builds ConcordanceTable internally.

        # perf:
        self._keyword_prev: str = ''
        self._results_cache: Union[pd.DataFrame, None] = None

    def set_keyword(self, keyword: str):
        self._keyword_prev = self.search_table.keyword
        if keyword == self.search_table.keyword:
            return self
        self.search_table.keyword = keyword
        return self

    def find(self) -> pd.DataFrame:
        if len(self.search_table.keyword) < 1:
            raise ValueError("Did you set the keyword? Call set_keyword()")

        if self._keyword_updated():
            self._results_cache = self.search_table._get_results()
        return self._results_cache

    def show_widget(self):
        return self.show()

    def _keyword_updated(self):
        return self._keyword_prev != self.search_table.keyword


if __name__ == '__main__':
    from juxtorpus.corpus import Corpus, CorpusBuilder

    import re

    tweet_wrapper = re.compile(r'([ ]?<[/]?TWEET>[ ]?)')

    builder = CorpusBuilder('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv')
    builder.set_text_column('processed_text')
    builder.set_nrows(100)
    builder.set_text_preprocessors([lambda text: tweet_wrapper.sub('', text)])
    builder.add_metas('tweet_lga', dtypes='category')
    builder.add_metas(['year', 'month', 'day'], dtypes='datetime')
    corpus = builder.build()
    corpus.summary()

    concordance = ATAPConcordance(corpus)
    concordance.set_keyword("MITIGATE").find()
    print()


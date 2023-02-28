""" Zipf
The Zipf's Law is an empirical law that says the rank-frequency distribution is an inverse
relation.

It first came about in quantitative linguistics, where given a corpus, the frequency of
any word is inversely proportional to its rank in the frequency table.
i.e. the second most frequent word is half of the most frequent word.
"""
import contextlib

import pandas as pd
import matplotlib.pyplot as plt
import logging

from juxtorpus.viz import Viz


class Zipf(Viz):
    def __init__(self, corpus):
        self._COL_COUNTS = 'counts'
        self._COL_WORD = 'word'
        self._COL_PROPORTIONS = 'proportions'
        self._COL_PROPORTIONS_ZIPF = 'zipf'

        self._df = self._build_dataframe(corpus,
                                         self._COL_COUNTS, self._COL_WORD,
                                         self._COL_PROPORTIONS, self._COL_PROPORTIONS_ZIPF)

        self._top = None

    def set_top(self, n: int):
        """ Set top n words to show. """
        self._top = n
        return self

    def render(self):
        df = self._df
        top = self._top if self._top is not None else len(df)

        if top > 25:
            logging.warning(
                f"Plotting {top} words may lag as there is a lot to render. Try a number < 25 with set_top()"
            )

        fig, ax = plt.subplots(figsize=(16, 10))
        ranks = [i for i in range(1, min(top + 1, len(df) + 1))]
        ax.plot(ranks, df[self._COL_PROPORTIONS_ZIPF].iloc[:top], color='gold', linewidth=2, label='zipf')
        ax.bar(ranks, df[self._COL_PROPORTIONS].iloc[:top], label='word')
        ax.set_xlabel("Rank")
        ax.set_ylabel("Proportion of max frequency")
        ax2 = ax.twiny()  # create another x-axis sharing y-axis - hence twin-'y'
        ax2.set_xticks(ticks=ranks, labels=df.iloc[:top].index)
        ax2.set_xlabel("Word")
        plt.xticks(fontsize=14, rotation=45)
        ax.legend()
        plt.show()

    @contextlib.contextmanager
    def remove_words(self, word_list):
        """ Allow for words to be temporarily removed from your plot without affecting the original."""
        _old_df = self._df.copy()
        _old_top = self._top
        try:
            self._df = self._df[~self._df.index.isin(word_list)]
            yield self
        finally:
            self._df = _old_df
            self._top = _old_top

    def _build_dataframe(self, corpus, col_count, col_word, col_proportions, col_zipf):
        df = corpus.dtm.freq_table(nonzero=True).series.sort_values(ascending=False)
        df.name = 'freq'
        df = df.to_frame()
        df = self._derive_proportions(df, col_proportions, col_zipf)
        return df

    def _derive_proportions(self, df, col_proportions, col_zipf):
        max_freq = df['freq'].max()
        df[col_proportions] = df.loc[:, 'freq'].apply(lambda c: c / max_freq)
        df[col_zipf] = [1 / i for i in range(1, len(df) + 1)]
        return df


if __name__ == '__main__':
    from pathlib import Path
    from juxtorpus.corpus import CorpusBuilder

    builder = CorpusBuilder(Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'))
    builder.set_text_column('processed_text')
    builder.set_nrows(100)
    corpus = builder.build()

    # optional - processed by spacy to use spacy matcher
    from juxtorpus.corpus.processors import SpacyProcessor
    import spacy

    nlp = spacy.blank('en')
    spacy_processor = SpacyProcessor(nlp)
    corpus = spacy_processor.run(corpus)

    zipf = Zipf(corpus)
    zipf.set_top(30)
    zipf.render()

    # temporarily remove words
    with zipf.remove_words(['the', 'to', 'of', 'a', 'is', 'and', 'you']) as zipff:
        zipff.render()

import pandas as pd
from typing import List, Tuple, Callable, Set, Dict
import matplotlib.pyplot as plt

from juxtorpus.viz import Viz

"""
PolarityBar is a Viz class that display the polarity of words and their scores given 2 sets.

It does not include shared words.

Stacked view: This does not assume anything but converts 1 set of scores to be negative.
Relative view: This displays the relative word dominance between the corpora.


Question:
should this visualisation expect positive and negatives?
OR
should this visualisation expect 2 sets of positives and do the negatives themselves?
"""


class PolarityBar(Viz):
    @staticmethod
    def from_(word_scores_A: List[Tuple[str, float]], word_scores_B: List[Tuple[str, float]],
              top: int = -1) -> 'PolarityBar':
        df_A = pd.DataFrame(word_scores_A, columns=['word', 'A_score']).set_index('word')
        df_B = pd.DataFrame(word_scores_B, columns=['word', 'B_score']).set_index('word')
        df = df_A.join(df_B, on=None, how='inner')  # inner = intersection only
        df['__summed__'] = df['A_score'] + df['B_score']
        df = df.sort_values(by='__summed__', ascending=False)
        if top > 0:
            df = df.iloc[:top]
        return PolarityBar(df)

    def __init__(self, word_score_df: pd.DataFrame):
        """ word_score_df expects a dataframe with a word column and 2 score columns. """
        self._df = word_score_df
        self._stacked_in_subplot: bool = False
        self._relative_in_subplot: bool = False

    def stack(self):
        """ Set up the bar visuals as stacked bars """
        fig, ax = plt.subplots()
        self._add_score_sum_to_df()
        _df = self._df.sort_values(by='AB_score')
        b1 = ax.barh(_df.index, _df['A_score'], color='green')
        b2 = ax.barh(_df.index, -_df['B_score'], color='red')

        plt.legend([b1, b2], ['CorpusA', 'CorpusB'], loc='upper right')
        plt.title("Stacked Frequency")
        return self

    def relative(self):
        """ Set up the bar visuals as a relative bar """
        fig, ax = plt.subplots()

        self._add_score_relative_to_df()
        _df = self._df.sort_values(by='AB_score_relative')

        _df['__viz__positive'] = _df['AB_score_relative'] > 0
        b = ax.barh(_df.index, _df['AB_score_relative'],
                    color=_df['__viz__positive'].map({True: 'g', False: 'r'}))

        # todo: legend does not display the correct colour
        plt.legend([b, b], ['CorpusA', 'CorpusB'], labelcolor=['green', 'red'])
        plt.title("Relative Frequency Differences")
        return self

    def _add_score_sum_to_df(self) -> str:
        if 'AB_score' not in self._df.columns:
            self._df['AB_score'] = self._df['A_score'] + self._df['B_score']
        return 'AB_score'

    def _add_score_relative_to_df(self) -> str:
        if 'AB_score_relative' not in self._df.columns:
            self._df['AB_score_relative'] = self._df['A_score'] + -self._df['B_score']
        return 'AB_score_relative'

    def render(self):
        # show the subplots
        plt.show()
        self._cleanup()

    def _cleanup(self):
        if '__viz__positive' in self._df.columns:
            self._df.drop(columns=['__viz__positive'], inplace=True)


if __name__ == '__main__':
    A = [
        ('hello', 0.25),
        ('there', 1.0),
    ]

    B = [
        ('good', 1.0),
        ('bye', 2.0),
        ('hello', 0.1)
    ]
    pbar = PolarityBar.from_(A, B)
    pbar.stack().render()
    pbar.relative().render()

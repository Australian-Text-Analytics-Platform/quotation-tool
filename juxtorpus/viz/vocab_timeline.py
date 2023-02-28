""" Vocabulary Timeline

Plots the vocabulary of a set of corpora on a timeline based on a date keys.


WIP
TODO - vocabulary is now randomly sampled from corpus. Should use term frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from juxtorpus.corpus import Corpus
from juxtorpus.viz import Viz

import random
from datetime import date


class VocabTimeline(Viz):
    def __init__(self, key_and_corpora_map: dict[date, Corpus]):
        self._map = key_and_corpora_map
        self._dates = self._map.keys()
        self._min_date = date(year=min(self._dates).year - 1, month=1, day=1)
        self._max_date = date(year=max(self._dates).year + 1, month=1, day=1)
        self._labels = self._generate_labels()

    def _generate_labels(self):
        _labels = list()
        for d, corpus in self._map.items():
            _labels.append(self._generate_label(d, corpus))
        return _labels

    def _generate_label(self, d: date, corpus: Corpus):
        return f"{d.year}\n" + ', '.join(random.sample(corpus.unique_words, 5))

    def _draw_hline(self, ax):
        ax.set_ylim(-2, 1.75)
        ax.set_xlim(self._min_date, self._max_date)
        ax.axhline(0, xmin=0.05, xmax=0.95, c='deeppink', zorder=1)

    def _draw_dots_on_line(self, ax):
        ax.scatter(self._dates, np.zeros(len(self._dates)), s=120, c='palevioletred', zorder=2)
        ax.scatter(self._dates, np.zeros(len(self._dates)), s=30, c='darkmagenta', zorder=3)

    def _position_alternating_labels(self, ax):
        label_offsets = np.zeros(len(self._dates))
        label_offsets[::2] = 0.35
        label_offsets[1::2] = -0.7
        for i, (l, d) in enumerate(zip(self._labels, self._dates)):
            ax.text(d, label_offsets[i], l, ha='center', fontfamily='serif', fontweight='bold',
                    color='royalblue', fontsize=12)

    def _draw_stems(self, ax):
        stems = np.zeros(len(self._dates))
        stems[::2] = 0.3
        stems[1::2] = -0.3
        markerline, stemline, baseline = ax.stem(self._dates, stems, use_line_collection=True)
        _ = plt.setp(markerline, marker=',', color='darkmagenta')
        _ = plt.setp(stemline, color='darkmagenta')

    def _hide_lines_on_border(self, ax):
        for spine in ["left", "top", "right", "bottom"]:
            _ = ax.spines[spine].set_visible(False)

    def _hide_tick_labels(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def render(self, height=16, width=8):
        if len(self._labels) != len(self._dates): raise RuntimeError("Misaligned number of labels and dates.")
        fig, ax = plt.subplots(figsize=(height, width), constrained_layout=True)
        self._draw_hline(ax)
        self._draw_dots_on_line(ax)
        self._position_alternating_labels(ax)
        self._draw_stems(ax)
        self._hide_lines_on_border(ax)
        self._hide_tick_labels(ax)
        plt.show()


if __name__ == '__main__':
    from juxtorpus.corpus import CorpusBuilder
    from pathlib import Path

    builder = CorpusBuilder(Path("./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv"))
    builder.add_metas(['year', 'month', 'day'], dtypes='datetime', lazy=True)
    builder.add_metas('tweet_lga', dtypes='category', lazy=True)
    builder.set_text_column('processed_text')
    corpus = builder.build()

    # now i want to be able to slice the corpus into different years.
    from juxtorpus.corpus import CorpusSlicer
    import pandas as pd
    from datetime import date

    slicer = CorpusSlicer(corpus)
    dt: pd.Timestamp

    timeline_map = {
        date(year=2019, month=1, day=1): slicer.filter_by_condition('datetime', lambda dt: dt.year == 2019),
        date(year=2020, month=1, day=1): slicer.filter_by_condition('datetime', lambda dt: dt.year == 2020)
    }

    vtimeline = VocabTimeline(timeline_map)
    vtimeline.render(height=16, width=8)

import math

import nltk

from rake_nltk import Rake
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Set, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import binarize
from scipy.sparse import csr_matrix
from collections import Counter
from spacy.matcher import Matcher
from spacy.tokens import Doc
import numpy as np

from juxtorpus.corpus import Corpus, SpacyCorpus
from juxtorpus.matchers import no_stopwords, is_word, no_puncs_no_stopwords
from juxtorpus.corpus.processors import SpacyProcessor


class Keywords(metaclass=ABCMeta):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    @abstractmethod
    def extracted(self) -> List[str]:
        raise NotImplemented("You are calling from the base class. Use one of the concrete ones.")


class RakeKeywords(Keywords):
    """ Implementation of Keywords extraction using Rake.
    package: https://pypi.org/project/rake-nltk/
    paper: https://www.researchgate.net/profile/Stuart_Rose/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents/links/55071c570cf27e990e04c8bb.pdf

    RAKE begins keyword extraction on a document by parsing its text into a set of candidate keywords.
    First, the document text is split into an array of words by the specified word delimiters.
    This array is then split into sequences of contiguous words at phrase delimiters and stop word positions.
    Words within a sequence are assigned the same position in the text and together are considered a candidate keyword.
    """

    def extracted(self):
        _kw_A = Counter(RakeKeywords._rake(sentences=self.corpus.texts().tolist()))
        return _kw_A.most_common(20)

    @staticmethod
    def _rake(sentences: List[str]):
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        r = Rake()
        r.extract_keywords_from_sentences(sentences)
        return r.get_ranked_phrases()


class TFIDFKeywords(Keywords):
    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        self.count_vec = CountVectorizer(
            tokenizer=TFIDFKeywords._do_nothing,
            preprocessor=TFIDFKeywords._preprocess,
            ngram_range=(1, 1)  # default = (1,1)
        )

    def extracted(self):
        corpus_tfidf = self._corpus_tf_idf(smooth=False)
        keywords = [(word, corpus_tfidf[0][i]) for i, word in enumerate(self.count_vec.get_feature_names_out())]
        keywords.sort(key=lambda w_tfidf: w_tfidf[1], reverse=True)
        return keywords
        # return TFIDFKeywords._max_tfidfs(self.corpus)

    def _corpus_tf_idf(self, smooth: bool = False):
        """ Term frequency is of the entire corpus. Idfs calculated as per normal. """
        tfs = self.count_vec.fit_transform(self.corpus.texts())
        idfs = binarize(tfs, threshold=0.99)
        if smooth:
            pass  # TODO: smoothing of idfs using log perhaps.
        return np.array(csr_matrix.sum(tfs, axis=0) / csr_matrix.sum(idfs, axis=0))

    @staticmethod
    def _max_tfidfs(corpus: Corpus):
        # get the tfidf score of the docs.
        # get the tfidf score of each word and rank them that way.
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus.texts())
        col_words = vectorizer.get_feature_names_out()
        max_tfidf_cols = [(col_words[i], X[:, i].max()) for i in range(X.shape[1])]
        max_tfidf_cols.sort(key=lambda t: t[1], reverse=True)
        return max_tfidf_cols

    @staticmethod
    def _do_nothing(doc):
        """ Used to override default tokenizer and preprocessors in sklearn transformers."""
        return doc

    @staticmethod
    def _preprocess(doc):
        """ Filters punctuations and normalise case."""
        return [doc[start:end].text.lower() for _, start, end in no_puncs_no_stopwords(nlp.vocab)(doc)]


class TFKeywords(Keywords):
    def __init__(self, corpus: SpacyCorpus):
        super(TFKeywords, self).__init__(corpus)
        if type(corpus) != SpacyCorpus:
            raise TypeError(f"TFKeywords requires {SpacyCorpus.__name__}. "
                            f"Please process it using {SpacyProcessor.__name__}.")

        self._vocab = corpus.nlp.vocab
        # config defaults
        self._threshold = -1
        self._normalise = True
        self._use_lemmas = True
        self._log = False

        self._df_min = 0.0
        self._df_max = 1.0

        self._filtered_words = dict()

    def set_max_term_freq_per_doc(self, threshold: int):
        self._threshold = threshold
        return self

    def normalise(self, to_normalise=True):
        """ Normalise by the number of words in the corpus. """
        self._normalise = to_normalise
        return self

    def log_freqs(self, to_log=True):
        """ Log the score from term frequencies. (Zip's law) """
        self._log = to_log
        return self

    def use_lemmas(self, use_lemmas=True):
        self._use_lemmas = use_lemmas
        return self

    def set_df_range(self, min_=0.0, max_=1.0):
        """ Set the document frequency range you want to include in the term frequencies.
        The values used here is normalised with the number of documents in the corpus.

        min_: value between 0.0 and max_.
        max_: value between min_ and 1.0.
        """
        if not 0.0 <= min_ < max_ <= 1.0:
            raise ValueError("Incorrect range. Must be 0.0 < min_ < max_ < 1.0")
        self._df_min = min_
        self._df_max = max_

    def extracted(self):
        word_freqs = self._count(self.corpus, normalise=self._normalise, log=self._log)
        return word_freqs

    def filtered(self):
        return self._filtered_words.copy()

    def _count(self, corpus: SpacyCorpus, normalise: bool, log: bool):
        doc_freq_counter = Counter()
        freq_counter = Counter()
        threshold_diff_to_adjust = 0

        _no_puncs_no_stopwords = no_puncs_no_stopwords(self._vocab)
        for d in corpus.docs():
            per_doc_freqs = dict()
            for token in self._find_matches(d, _no_puncs_no_stopwords, lemma_only=self._use_lemmas):
                current = per_doc_freqs.get(token, 0)
                per_doc_freqs[token] = current

            # apply threshold here and count the difference.
            for k, v in per_doc_freqs.items():
                _orig_value = v
                per_doc_freqs[k] = max(v, self._threshold)
                threshold_diff_to_adjust += _orig_value - self._threshold
            freq_counter.update(per_doc_freqs)
            # set a max on per_doc_freqs to 1 and add to doc_freq_counter
            doc_freq_counter.update({k: 1 for k, _ in per_doc_freqs.items()})

        # remove the words
        self._filtered_words = dict()
        for k, doc_freq in doc_freq_counter.items():
            doc_freq_norm = doc_freq / len(corpus)
            if doc_freq_norm < self._df_min:
                del freq_counter[k]
                self._filtered_words[k] = f"Minimum doc freq threshold exceeded. {doc_freq_norm} < {self._df_min}."
            elif doc_freq_norm >= self._df_max:
                del freq_counter[k]
                self._filtered_words[k] = f"Maximum doc freq threshold exceeded. {doc_freq_norm} > {self._df_max}."
            else:
                continue

        num_words = corpus.num_terms - threshold_diff_to_adjust
        freq_counter = dict(freq_counter)
        if log:
            for k in freq_counter.keys():
                freq_counter[k] = math.log(freq_counter.get(k))
            num_words = math.log(num_words)
        if normalise:
            for k in freq_counter.keys():
                freq_counter[k] = (freq_counter.get(k) / num_words) * 100
        return sorted(freq_counter.items(), key=lambda kv: kv[1], reverse=True)

    def _find_matches(self, doc: Doc, matcher: Matcher, lemma_only: bool):
        for _, start, end in matcher(doc):
            span = doc[start:end]
            if lemma_only:
                yield span.lemma_.lower()
            else:
                yield span.text.lower()


if __name__ == '__main__':
    from juxtorpus.corpus import Corpus, CorpusBuilder
    import re

    tweet_wrapper = re.compile(r'([ ]?<[/]?TWEET>[ ]?)')

    builder = CorpusBuilder('/Users/hcha9747/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv')
    builder.set_text_column('text')
    builder.set_nrows(10)
    builder.set_text_preprocessors([lambda text: tweet_wrapper.sub('', text)])
    corpus = builder.build()

    from juxtorpus.corpus.processors import SpacyProcessor
    import spacy

    nlp = spacy.load('en_core_web_sm')
    spacy_processor = SpacyProcessor(nlp)
    corpus = spacy_processor.run(corpus)

    # TF Keywords
    tf = TFKeywords(corpus)
    tf.set_max_term_freq_per_doc(3).normalise().log_freqs(False)
    tf.set_df_range(0.01, 0.5)
    print('\n'.join((str(x) for x in tf.extracted()[:10])))
    print('\n'.join((str(x) for x in tf.filtered().items())))

    tf = TFKeywords(corpus)
    tf.set_max_term_freq_per_doc(3).normalise().log_freqs(False).use_lemmas(True)
    lemmas = tf.extracted()[:10]
    tf.set_max_term_freq_per_doc(3).normalise().log_freqs(False).use_lemmas(False)
    no_lemmas = tf.extracted()[:10]

    import pandas as pd

    print(pd.concat([pd.DataFrame(lemmas, columns=['lemmas', 'score']),
                     pd.DataFrame(no_lemmas, columns=['not_lemma', 'score'])], axis=1))

    print()

"""
Spacy Matchers
https://spacy.io/usage/rule-based-matching

NOTE: Each dictionary represents 1 token.
"""

from typing import Dict
from spacy.matcher import Matcher
from spacy import Vocab

# TODO: rework this into PATTERNS only - not the Matcher, since its global, we can't control its state.
custom_hashtags_atap = {"_": {
    "hashtag": "#atap"
}}
custom_hashtags_atap_regex = {"_": {
    "hashtag": {"REGEX": r'^#[Aa][Tt][Aa][Pp]'}
}}

is_noun = {"POS": "NOUN"}
is_verb = {"POS": "VERB"}


def hashtags(vocab: Vocab):
    _hashtags = Matcher(vocab)
    _hashtags.add("hashtags", patterns=[
        [{"TEXT": "#"}, {"IS_ASCII": True}]
    ])
    return _hashtags


def at_mentions(vocab: Vocab):
    _at_mentions = Matcher(vocab)
    _at_mentions.add("mentions", patterns=[
        [{"TEXT": {"REGEX": r"^@[\S\d]+"}}]
    ])
    return _at_mentions


def urls(vocab: Vocab):
    _urls = Matcher(vocab)
    _urls.add("urls", patterns=[
        [{"LIKE_URL": True}]
    ])
    return _urls


def is_word(vocab: Vocab):
    _no_puncs = Matcher(vocab)
    _no_puncs.add("no_punctuations", patterns=[
        [{"IS_PUNCT": False, 'IS_ALPHA': True}]
    ])
    return _no_puncs


def no_puncs_no_stopwords(vocab: Vocab):
    m = Matcher(vocab)
    m.add("no_punc_or_sw", patterns=[
        [{"IS_PUNCT": False, "IS_STOP": False, "IS_ALPHA": True}]
    ])
    return m


def no_stopwords(vocab: Vocab):
    _no_stopwords = Matcher(vocab)
    _no_stopwords.add("no_stopwords", patterns=[
        [{"IS_STOP": False}]
    ])
    return _no_stopwords

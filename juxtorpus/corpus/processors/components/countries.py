import requests
from spacy.tokens import Token, Span
from spacy.matcher import PhraseMatcher

"""
Sample code of pipeline extension component from spaCy.
https://explosion.ai/blog/spacy-v2-pipelines-extensions
"""


class Countries(object):
    name = 'countries'  # component name shown in pipeline

    def __init__(self, nlp, label="GPE"):
        # request all country data from the API
        r = requests.get("https://restcountries.eu/rest/v2/all")
        self.countries = {c['name']: c for c in r.json()}  # create dict for easy lookup
        # initialise the matcher and add patterns for all country names
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add("COUNTRIES", None, *[nlp(c) for c in self.countries.keys()])
        self.label = nlp.vocab.strings[label]  # get label ID from vocab
        # register extensions on the Token
        Token.set_extension("is_country", default=False)
        Token.set_extension("country_capital")
        Token.set_extension("country_latlng")

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        for _, start, end in matches:
            # create Span for matched country and assign label
            entity = Span(doc, start, end, label=self.label)
            spans.append(entity)
            for token in entity:  # set values of token attributes
                token._.set("is_country", True)
                token._.set("country_capital", self.countries[entity.text]["capital"])
                token._.set("country_latlng", self.countries[entity.text]["latlng"])
        doc.ents = list(doc.ents) + spans  # overwrite doc.ents and add entities – don't replace!
        for span in spans:
            span.merge()  # merge all spans at the end to avoid mismatched indices
        return doc  # don't forget to return the Doc!

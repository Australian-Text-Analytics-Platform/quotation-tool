from spacy import Language
from spacy.tokens import Doc
from spacy.matcher import Matcher

from juxtorpus.matchers import at_mentions
from juxtorpus.corpus.processors.components import Component

class MentionsComp(Component):
    def __init__(self, nlp: Language, name: str, attr: str):
        super(MentionsComp, self).__init__(nlp, name, attr)
        if Doc.has_extension(self._attr):
            raise KeyError(f"{self._attr} already exists. {MentionsComp.__name__} will not function properly.")
        Doc.set_extension(self._attr, default=[])

        self.matcher = at_mentions(nlp.vocab)

    def __call__(self, doc: Doc) -> Doc:
        for _, start, end in self.matcher(doc):
            span = doc[start: end]
            getattr(getattr(doc, '_'), self._attr).append(span.text)
        return doc


if __name__ == '__main__':
    import spacy

    nlp = spacy.blank('en')

    doc = nlp("The Top HSC STUDENT WAS A CLIMATE STRIKER! @ScottMorrisonMP @GladysB\n")
    from juxtorpus.matchers import at_mentions
    matcher = at_mentions(nlp.vocab)
    for _, start, end in matcher(doc):
        span = doc[start: end]
        print(span)
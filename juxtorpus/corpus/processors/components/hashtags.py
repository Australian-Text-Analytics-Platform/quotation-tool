from spacy import Language
from spacy.tokens import Doc
from spacy.matcher import Matcher

from juxtorpus.corpus.processors.components import Component


class HashtagComponent(Component):
    def __init__(self, nlp: Language, name: str, attr: str):
        super().__init__(nlp, name, attr)
        self._getter = lambda hashtags: [ht.text for ht in hashtags]
        if Doc.has_extension(self._attr):
            raise KeyError(f"{self._attr} already exists. {HashtagComponent.__name__} will not function properly.")
        Doc.set_extension(self._attr, default=[])  # doc._.hashtags is may now be accessed.
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("hashtag", patterns=[
            [{"TEXT": "#"}, {"IS_ASCII": True}]
        ])

    def __call__(self, doc: Doc) -> Doc:
        for _, start, end in self.matcher(doc):
            span = doc[start: end]
            getattr(getattr(doc, '_'), self._attr).append(span.text)
        return doc


if __name__ == '__main__':
    # using the custom component...
    @Language.factory('extract_hashtags')
    def create_hashtag_component(nlp: Language, name: str):
        return HashtagComponent(nlp, name, 'hashtags')


    import spacy

    nlp = spacy.blank('en')
    nlp.add_pipe('extract_hashtags')
    doc = nlp("The #MarchForLife is so very extremely important. To all of you marching --- you have my full support!")
    print(f"doc._.hashtags: {doc._.hashtags}")

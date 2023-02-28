"""
SpaCy Custom Components

This package contains subclasses of Component. These subclasses are custom spaCy components that you create and
can be added to the nlp pipeline.

Documentations:
https://spacy.io/usage/processing-pipelines#custom-components
https://explosion.ai/blog/spacy-v2-pipelines-extensions
https://spacy.io/api/language
"""
import abc
from typing import Dict, Union, Callable
from abc import ABCMeta, abstractmethod

from spacy.tokens.doc import Doc
from spacy.language import Language


class Component(metaclass=ABCMeta):
    def __init__(self, nlp: Language, name: str, attr: str):
        self._nlp = nlp
        self._name = name
        self._attr = attr

    @property
    def attr(self):
        return self._attr

    # todo: see spaCy Language doc, nlp.pipe(as_tuples=True) - this allow context to be passed down i believe.
    @abstractmethod
    def __call__(self, doc: Doc) -> Doc:
        """ Operations of this component to modify spaCy Doc object is performed here. """
        raise NotImplementedError()


class ComponentImpl(Component):

    def __init__(self, nlp: Language, name: str, a_setting: bool):
        super(ComponentImpl, self).__init__(nlp, name)
        print(f"{self.__init__}: {name}, {a_setting}")

    def __call__(self, doc: Doc):
        print(f"{self.__call__}: Processing doc: {doc}...")
        return doc


"""Examples custom components:

stateless: use @Language.component()
stateful: use @Language.factory()
"""


@Language.component(name="stateless_custom_component")
def stateless_custom_component(doc: Doc) -> Doc:
    print("I don't do anything but I demonstrate a stateless component implementation.")
    return doc


# EACH FACTORY wrapper represents an INSTANCE of the stateful component. The instance name is given by the name arg.
@Language.factory(name='stateful_custom_component', default_config={"a_setting": True})
def stateful_custom_component(nlp: Language, name: str, a_setting: bool) -> Callable[[Doc], Doc]:
    return ComponentImpl(nlp, name, a_setting)


if __name__ == '__main__':
    import spacy

    nlp = spacy.blank('en')
    nlp.add_pipe("stateless_custom_component")
    nlp.add_pipe("stateful_custom_component", config={"a_setting": False})  # Notice default setting is True.
    print(nlp("This is a sentence."))
    print(nlp.pipe_names)

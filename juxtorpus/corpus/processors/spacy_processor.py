""" Spacy Pipelines

A collection of spacy pipeline components for different use cases.
This will improve the efficiency of the library. Efficiency are compute time and memory usage.

Reduce Memory usage:
Spacy allows you to 'include' and exclude components. The tradeoff is that the spacy nlp model needs to be reloaded.

Improve computation/inference time:
Spacy allows you to 'disable'/'enable' components in your models. This skips parts of the pipeline hence reduces
computation time. But it does not save memory.

https://spacy.io/usage/processing-pipelines


Things to note:
The quick and easy way to use a pretrained pipeline from spacy is via spacy.load('...'), this creates a Language object
configured with their config.cfg and load a bunch of models from binaries.
(You may go to spacy.utils - load_model_from_init_py, load_model_from_path if you want to dig deeper)

Design:
Assuming we use an out-of-the-box pipeline from spaCy, the models are going to be loaded to RAM from disk.
What this means is that we can't just call 'nlp.remove_pipe' then 'nlp.add_pipe' on an out-of-the-box component.
If we do, we need to load the model back on ourselves with the config from the package,
i.e. call from_disk that component (note: some components inherit from_disk directly from TrainablePipe class).
This is possible to do/hack if we wrap some logic around spacy.load(...exclude=[]), just exclude everything else and
repopulate the pipeline. But for our case, I think I'll just keep it simple for now.
What this means is in terms of the design:
Out-of-the-box components are 'disabled' ONLY when not in use.
Custom components are 'added' and 'removed' from our pipelines as we see fit. These are appended.

Future:
1. This simple behaviour is a tech debt. If we have our own models later and write our own config.cfg + load models from
disk, it'll be a problem; Both with the current behaviour and the RAM usage.
Trainable components need to override from_disk() method to load the models.

2. Ordering of components are not incorporated. (todo: add an 'after' and 'before' attribute to the Components? - see nlp.analyze_pipes)


Other resources:
https://spacy.io/models#design
https://spacy.io/api/language#select_pipes

Notes:
    + use 'senter' instead of 'parser' when dependency parsing is not required. Inference is ~10x faster.
"""

from spacy import Language
from datetime import datetime
import pandas as pd
from tqdm.auto import tqdm

from juxtorpus.corpus import Corpus, SpacyCorpus
from juxtorpus.corpus.processors import Processor, ProcessEpisode
from juxtorpus.corpus.processors.components import Component
from juxtorpus.corpus.processors.components.hashtags import HashtagComponent
from juxtorpus.corpus.processors.components.mentions import MentionsComp
from juxtorpus.corpus.processors.components.sentiment import SentimentComp
from juxtorpus.corpus.meta import DocMeta

import colorlog

logger = colorlog.getLogger(__name__)


@Language.factory("extract_hashtags")
def create_hashtag_component(nlp: Language, name: str):
    return HashtagComponent(nlp, name, attr='hashtags')


@Language.factory("extract_mentions")
def create_mention_component(nlp: Language, name: str):
    return MentionsComp(nlp, name, attr='mentions')


@Language.factory("extract_sentiments")
def create_sentiment(nlp: Language, name: str):
    return SentimentComp(nlp, name, attr='sentiment')


class SpacyProcessor(Processor):
    """ SpacyProcessor
    This class processes a Corpus object into a SpacyCorpus.
    It takes in spacy's `Language` model and uses it to process the texts in the Corpus and then set up the
    same metadata in the new SpacyCorpus.
    """
    built_in_component_attrs = {
        'ner': 'ents'
    }

    def __init__(self, nlp: Language):
        self._nlp = nlp

    @property
    def nlp(self):
        return self._nlp

    def _process(self, corpus: Corpus) -> SpacyCorpus:
        start = datetime.now()
        logger.info(f"Processing corpus of {len(corpus)} documents...")
        texts = corpus.texts()
        # doc_generator = (doc for doc in tqdm(self.nlp.pipe(texts)))
        doc_generator = self.nlp.pipe(texts)
        docs = pd.Series(doc_generator, index=texts.index)
        logger.info("Done.")
        logger.debug(f"Elapsed time: {datetime.now() - start}s.")
        return SpacyCorpus.from_corpus(corpus, docs, self.nlp)

    def _add_metas(self, corpus: Corpus):
        """ Add the relevant meta-objects into the Corpus class.

        Note: attribute name can come from custom extensions OR spacy built in. see built_in_component_attrs.
        """
        for name, comp in self.nlp.pipeline:
            _attr = comp.attr if isinstance(comp, Component) else self.built_in_component_attrs.get(name, None)
            if _attr is None: continue
            generator = corpus._df.loc[:, corpus.COL_TEXT]
            meta = DocMeta(id_=name, attr=_attr, nlp=self.nlp, docs=generator)
            corpus._meta_registry[meta.id] = meta

    def _create_episode(self) -> ProcessEpisode:
        return ProcessEpisode(
            f"Spacy Processor processed on {datetime.now()} with pipeline components {', '.join(self.nlp.pipe_names)}."
        )


if __name__ == '__main__':
    import pathlib
    from juxtorpus.corpus import CorpusBuilder, CorpusSlicer

    builder = CorpusBuilder(
        pathlib.Path('/Users/hcha9747/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv')
    )
    builder.set_nrows(10000)
    builder.set_text_column('text')
    corpus = builder.build()

    # Now to process the corpus...

    import spacy

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('extract_hashtags')

    spacy_processor = SpacyProcessor(nlp)
    corpus = spacy_processor.run(corpus)

    print(corpus.history())

    # Now to test with corpus slicer...

    slicer = CorpusSlicer(corpus)
    slice = slicer.filter_by_condition('extract_hashtags', lambda x: '#RiseofthePeople' in x)
    print(len(slice))

    slice = slicer.filter_by_item('extract_hashtags', '#RiseofthePeople')
    print(len(slice))

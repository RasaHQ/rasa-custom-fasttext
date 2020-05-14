<img src="square-logo.svg" width=200 height=200 align="right">

This repository contains a demo project with a custom made tokenizer and featurizer.

It is maintained by Vincent D. Warmerdam, Research Advocate as [Rasa](https://rasa.com/).

# Implementing Fasttext for Rasa 

Rasa offers many useful components to build a digital assistant but 
sometimes you may want to write your own. This document is part
of a series where we will create increasingly complex components from
scratch. In this document we'll build a component that can add
fasttext word embeddings to your Rasa pipeline.

## Example Project

You can clone the repository found [here](https://github.com/RasaHQ/rasa-custom-fasttext) 
if you'd like to be able to run the same project. The repository contains a relatively small 
rasa project; we're only dealing with four intents and one entity. It's very similar to the project
that was discussed in the previous guide. 

### `data/nlu.md`

```md
## intent:greet
- hey
- hello
...

## intent:goodbye
- bye
- goodbye
...

## intent:bot_challenge
- are you a bot?
- are you a human?
...

## intent:talk_code
- i want to talk about python
- What does this javascript error mean? 
...
```

### `data/stories.md`

```md
## just code
* talk_code
  - utter_proglang

## check bot
* bot_challenge
  - utter_i_am_bot
* goodbye
  - utter_goodbye

## hello and code
* greet
    - utter_greet
* talk_code
    - utter_proglang
```

We'll use the config.yml file that we had before, that means that we have our 
`printer.Printer` at our disposal too. 

## Fasttext Features 

The goal of this document is to create custom a component that adds word embeddings
from fasttext to Rasa. What's nice about these embeddings is they're available for 
[157 languages](https://fasttext.cc/docs/en/crawl-vectors.html#models) and the 
[fasttext library](https://fasttext.cc/docs/en/unsupervised-tutorial.html) also offers 
an option to train your own. We won't go into the details of how fasttext is trained 
but our algorithm whiteboard playlist does offer some details on how the
[CBOW and Skipgram](https://www.youtube.com/watch?v=BWaHLmG1lak) algorithms work. 

## Implementation 

Fasttext offers a simple python interface which really helps with the implementation. 
There's a downside to fasttext embeddings though; they are huge. The english vectors
,uncompressed, are about 7.5Gb on disk. If these were to go into a huge zip file then 
you'd end up with a model file that's too big to upload to a docker container. 

Instead of making a component that will persist these embeddings we'll make a component
that requires the presense of a cached directory. This way you could mount a disk for
your docker container that contains these embeddings without having to deal with bloated
docker containers. Implementation wise that also means that you'll need to add 
parameters that are required for the component. 

We're written an implementation of this into a file called `ftfeats.py`. The contents of
it are shown below. 

### `ftfeats.py`

```python
import typing
from typing import Any, Optional, Text, Dict, List, Type
import fasttext
import numpy as np
import os

from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Tokenizer

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata
from rasa.nlu.constants import DENSE_FEATURE_NAMES, DENSE_FEATURIZABLE_ATTRIBUTES, TEXT


class FastTextFeaturizer(DenseFeaturizer):
    """A new component"""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["fasttext"]

    defaults = {"file": None, "cache_dir": None}
    language_list = []

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        path = os.path.join(component_config["cache_dir"], component_config["file"])
        self.model = fasttext.load_model(path)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_fasttext_features(example, attribute)

    def set_fasttext_features(self, message: Message, attribute: Text = TEXT):
        text_vector = self.model.get_word_vector(message.text)
        word_vectors = [
            self.model.get_word_vector(t.text)
            for t in message.data["tokens"]
            if t.text != "__CLS__"
        ]
        X = np.array(word_vectors + [text_vector])  # remember, we need one for __CLS__

        features = self._combine_with_existing_dense_features(
            message, additional_features=X, feature_name=DENSE_FEATURE_NAMES[attribute]
        )
        message.set(DENSE_FEATURE_NAMES[attribute], features)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.set_fasttext_features(message)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        else:
            return cls(meta)
```

There's a few things to point out in general about this component. 

1. The `FastTextFeaturizer` inherits from `rasa.nlu.featurizers.featurizer.DenseFeaturizer`.
2. The implementation depends on the `fasttext` python library and it requires that a `Tokenizer` is present in the pipeline.
3. There are two default parameters which are set to `None`; `file` and `cache_dir`. The idea is that `cache_dir` needs to
point to a (mounted) directory that contains the fasttext embeddings and `file` can needs to point to the name of the file
in that folder that contains the embeddings. We load the `fasttext` embeddings during initialisation of the object. 
4. Because the fasttext embeddings are available in many languages we don't check for the presence of a language setting. 
5. The `persist` and `load` methods are unimplemented because we do not intend on saving/loading the embeddings ourselves.
They should only be loaded from the cached directory. 

The actual work in this components occurs in the `set_fasttext_features` method 
so we'll zoom in on that below. 

```python
def set_fasttext_features(self, message: Message, attribute: Text = TEXT):
    # we first have fasttext translate the entire sentence 
    text_vector = self.model.get_word_vector(message.text)
    # next we have fasttext translate each token individually *except* the __CLS__ token
    word_vectors = [
        self.model.get_word_vector(t.text)
        for t in message.data["tokens"]
        if t.text != "__CLS__"
    ]

    # we combine everything together 
    # remember, the text_vector for the entire sentence should match the __CLS__ token
    X = np.array(word_vectors + [text_vector])

    # here we use `._combine_with_existing_dense_features` from `DenseFeaturizer`
    # to make sure we never overwrite the features that are already there. 
    features = self._combine_with_existing_dense_features( z
        message, additional_features=X, feature_name=DENSE_FEATURE_NAMES[attribute]
    )
    # finally we set these new features unto our message for our pipeline
    message.set(DENSE_FEATURE_NAMES[attribute], features)
```

You'll notice that this `set_fasttext_features` method is used both in the `train` method 
(because we have to prepare the training data) and during the `process` step. 

## Demo 

We're going to be using out `printer.Printer` component from a previous tutorial to
demonstrate the effect of this component. This is what the pipeline in our `config.yml` looks like; 

```yml
language: en

pipeline:
- name: printer.Printer
  alias: start
- name: tbfeats.TextBlobTokenizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 3
- name: ftfeat.FastTextFeaturizer
  cache_dir: "/path/to/vectors/"
  file: "cc.en.300.bin"
- name: printer.Printer
  alias: after fasttext
- name: DIETClassifier
  epochs: 20
```

Note that we're keeping the number of epochs low here because we're only
interested in seeing the effect from our own custom component. Note that the 
file that we're giving to our `ftfeat.FastTextFeaturizer` was downloaded from
[here](https://fasttext.cc/docs/en/crawl-vectors.html#models). Don't forget to unzip!

If you were to run this pipeline, here's what you'll get; 

```
> rasa train; rasa shell 
> hello how are you
after fasttext
text : hello how are you
intent: {'name': None, 'confidence': 0.0}
entities: []
tokens: ['hello', 'how', 'are', 'you', '__CLS__']
text_sparse_features: <5x420 sparse matrix of type '<class 'numpy.longlong'>'
        with 98 stored elements in COOrdinate format>
text_dense_features: Dense array with shape (5, 300)
```

There's a few things to note. 

1. The tokens give us 4 word tokens and one `__CLS__` token. This `__CLS__` token 
represents the entire sentence as opposed to just words. 
2. You can see that the sparse features that are generated have a shape (5x420). That means
that we have 420 features for each of the 5 tokens. 
3. You can see that the dense features, generated by our own custom component, generate
300 floating number features for each of the 5 tokens too. 

So there you have it, fasttext is now giving features to the model. 

## Conclusion 

This document demonstrates how you are able to add fasttext embeddings to your
pipeline by building a custom component. In practice you'll need to be very mindful
of the disk space needed for these embeddings. But we hope this guide makes it easier
for you to experiment with word embeddings. We're especially interested in hearing 
if these features make a difference in non-english languages.
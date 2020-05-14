<img src="square-logo.svg" width=200 height=200 align="right">

This repository contains a demo project with a custom made tokenizer and featurizer.

It is maintained by Vincent D. Warmerdam, Research Advocate as [Rasa](https://rasa.com/).

# Two Spelling Components
## A Tokenizer and A Featurizer

Rasa offers many useful components to build a digital assistant but 
sometimes you may want to write your own. This document is part
of a series where we will create increasingly complex components from
scratch. In this document we'll build a component that can add
fasttext word embeddings to your Rasa pipeline.

## Example Project

You can clone the repository found [here](https://github.com/RasaHQ/rasa-custom-spelling-featurizer) 
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

To add the fasttext features we'll 
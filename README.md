# Negation detection of concepts in Dutch clinical text

This repository contains the source code for a Dutch negation detector for clinical text developed in the scope of the  
[ACCUMULATE](https://github.com/clips/accumulate) project. The negation detection is performed specifically for detected clinical concepts within a sentence, rather than on the token level.

## Requirements

* Python 3
* [Frog](https://languagemachines.github.io/frog/)

## Usage

### Data preprocessing

This module processes raw clinical text using Frog and integrates the preprocessed output with user-provided concept annotations on the raw text.
Gold standard negation annotations can be included for later evaluation.

```
from preprocessing import PreprocessCorpus
    
preprocessor = PreprocessCorpus()
preprocessed_instances = preprocessor(file_ids)
# file_ids = list of paths to .json files containing one dictionary each with the relevant input data
    
# example input dictionary:
# input_dictionary['text'] = raw clinical text to be processed by Frog
# input_dictionary['concept_spans'] = [{'begin': start_idx, 'end': end_index},
                                       {'begin': start_idx, 'end': end_index}]                           
# if gold standard annotations are present for negation:
# input_dictionary['negation_status'] = [True, False]
```

### Tagging of negation cues

```
from negation_tagger import NegationTagger
    

tagger = NegationTagger()
tagged_sentences = tagger(preprocessed_instances)
```

### Negation detection of clinical concepts

```
from negation_detector import NegationDetector, NegationDetectorEvaluation
    

# usage for data WITHOUT gold standard negation annotations
detector = NegationDetector()
instances_detection_data = detector.detect(preprocessed_instances, model)
# choose model from ['baseline', 'dependency', 'ensemble']
                                 

# usage for data WITH gold standard negation annotations
detector = NegationDetectorEvaluation()
results = detector(preprocessed_instances, model)
```

##### Baseline model

The baseline model only has the surface information of the sentence to work with, i.e., the tokens along with all concept and cue tags assigned to these tokens. The system matches all cues with the immediately following concept.

##### Dependency model

This model uses a dependency parse to match cues with concepts, and therefore does not rely on the surface word order to match cues with concepts. Because of this insensitivity to the surface word order, the model is able to consider the contextual relationship between words within a sentence to decrease false positives in comparison to linguistically agnostic models such as the baseline model. This notion has been exploited in [previous work on negation detection of concepts in English clinical text.](https://www.ncbi.nlm.nih.gov/pubmed/25791500) In contrast with the baseline model, which matches every negation cue with at most one concept, the dependency model is able to capture conjunctions between concepts which are negated with the same single cue, since such conjunctions are reflected in the dependency parse.

##### Ensemble model

This model combines the predictions of the baseline and dependency models in a way which highlights their strengths. While the dependency model suffers from unpredictable false positives whenever error percolation from the dependency parser arises, the false positives of the baseline model result from a transparent bias (matching any following concept in the sentence, regardless of its distance to the negation cue) which can be remedied by the increased precision of the dependency model. Therefore, we take the output of the baseline model as initial input, and replace the baseline predictions for the predicted negated concepts by the dependency predictions for these concepts. In this way, we avoid the unpredictable false positives of the dependency model while addressing the false positives of the baseline model.

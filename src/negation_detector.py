from collections import defaultdict
import json
from negation_triggers import negation_triggers


class NegationDetector:

    def __init__(self):

        self.negation_cues = negation_triggers['pre']
        self.allowed_dependency_steps = 1
        self.fuse_conjuncted = True

        self.evaluation_mode = False

    def load_instances(self, path_to_instances):

        with open(path_to_instances, 'r') as f:
            sentence_instances, gold = json.load(f)

        if self.evaluation_mode:
            if not gold:
                raise ValueError('If NegationDetector used in evaluation mode, loaded data should be gold data!')

        return sentence_instances

    def detect(self, sentence_instances, model):

        all_detection_data = []
        for sentence_instance in sentence_instances:
            detection_data = self.detect_instance(sentence_instance, model)
            all_detection_data += detection_data

        return all_detection_data

    def detect_instance(self, sentence_instance, model):

        if model == 'ensemble':
            detection_data = self.detect_negation_ensemble(sentence_instance)
        elif model in ['baseline', 'dependency']:
            detection_data = self.detect_negation(sentence_instance, model)
        else:
            raise ValueError('{} is no valid model, choose between baseline, dependency and ensemble'.format(model))

        return detection_data

    def detect_negation(self, instance_data, model, use_for_ensemble=False):

        # unpack data
        instance, instance_id = instance_data
        tags, metadata, sentence = instance
        tokens, spans, dependency_data = list(zip(*sentence))

        # extract valid modality-concept matches
        valid_modality_concept_matches = self.extract_valid_modality_concept_matches(tags)

        # extract concept idxs for reference
        concept_tagsets = self.extract_concept_tagsets(tags)

        # match concepts with modality
        if model == 'baseline':
            matched_concepts = self.baseline_detector_premodification(valid_modality_concept_matches, concept_tagsets,
                                                                      metadata)
        elif model == 'dependency':
            matched_concepts = self.dependency_detector(valid_modality_concept_matches, concept_tagsets, metadata,
                                                        dependency_data)
        else:
            raise ValueError('{} is not a valid model, choose between baseline and dependency')

        concepts = sorted(concept_tagsets.keys(), key=lambda x: int(x[1:]))
        detection_data = [matched_concepts, concepts, metadata, instance_id]

        # fuse predictions for conjuncted concepts
        if not use_for_ensemble:
            if model == 'dependency':
                if self.fuse_conjuncted:
                    detection_data = self.detect_conjunct_negation(instance_data, detection_data)

        return detection_data

    def detect_negation_ensemble(self, instance):

        # collect predictions of both models
        baseline_detection_data = self.detect_negation(instance, 'baseline', use_for_ensemble=True)
        dependency_detection_data = self.detect_negation(instance, 'dependency', use_for_ensemble=True)

        baseline_matched_concepts = baseline_detection_data[0]
        dependency_matched_concepts = dependency_detection_data[0]

        # fuse predictions of both model
        ensembled_matched_concepts = self.ensemble_predictions(baseline_matched_concepts, dependency_matched_concepts)

        # fuse with other data
        ensembled_detection_data = list(baseline_detection_data[1:])
        ensembled_detection_data.insert(0, ensembled_matched_concepts)

        # fuse predictions for conjuncted concepts
        if self.fuse_conjuncted:
            ensembled_detection_data = self.detect_conjunct_negation(instance, ensembled_detection_data)

        return ensembled_detection_data

    def detect_conjunct_negation(self, instance, detection_data):
        conjuncted_concepts = self.extract_conjunct_concepts(instance)
        conjuncted_detection_data = self.fuse_conjunct_concept_modalities(conjuncted_concepts, detection_data)

        return conjuncted_detection_data

    @staticmethod
    def fuse_conjunct_concept_modalities(conjuncted_concepts, detection_data):
        matched_concepts = detection_data[0]
        concepts = detection_data[1]

        for concept in concepts:
            if not matched_concepts[concept]:
                for conjuncted_concept in conjuncted_concepts[concept]:
                    conjuncted_concept_modality = matched_concepts[conjuncted_concept]
                    matched_concepts[concept] = conjuncted_concept_modality

        detection_data[0] = matched_concepts

        return detection_data

    def extract_conjunct_concepts(self, instance):

        # check if concepts are conjuncted by checking a concept for any conjunction relation with other concepts in
        # the sentence, then tie their faiths together!

        instance, instance_id = instance
        tags, metadata, sentence = instance
        tokens, spans, dependency_data = list(zip(*sentence))

        # extract direct conjunction dependencies from dependency data
        conjunction_dependencies = defaultdict(set)
        for token_index, (dependency_relation, dependency_idx) in enumerate(dependency_data):
            if dependency_relation == 'cnj':
                # see what other cnj tokens depend on the same conjuncting head
                head_conjuncting_idx = int(dependency_idx) - 1
                conjunction_dependencies[head_conjuncting_idx].add(token_index)

        concept_tagsets = self.extract_concept_tagsets(tags)
        inverted_concept_tagsets = defaultdict(set)
        for concept, idxs in concept_tagsets.items():
            for idx in idxs:
                inverted_concept_tagsets[idx].add(concept)

        conjuncted_concept_sets = []
        for conjuncted_tokens_idxs in conjunction_dependencies.values():
            conjuncted_concept_set = set()
            for token_idx in conjuncted_tokens_idxs:
                token_idx_concepts = inverted_concept_tagsets[token_idx]
                if not token_idx_concepts:
                    continue
                if len(token_idx_concepts) > 1:
                    # take longest concept
                    token_idx_concept_lens = {concept: len(concept_tagsets[concept]) for concept in token_idx_concepts}
                    token_idx_concept = max(token_idx_concept_lens.items(), key=lambda x: x[1])[0]
                else:
                    token_idx_concept = next(iter(token_idx_concepts))
                conjuncted_concept_set.add(token_idx_concept)
            conjuncted_concept_sets.append(conjuncted_concept_set)

        conjuncted_concepts = defaultdict(set)
        for conjuncted_concept_set in conjuncted_concept_sets:
            for conjuncted_concept in conjuncted_concept_set:
                conjuncted_concepts[conjuncted_concept].update(conjuncted_concept_set)

        conjuncted_concepts = {k: {x for x in v if x != k} for k, v in conjuncted_concepts.items()}
        conjuncted_concepts = defaultdict(set, conjuncted_concepts)  # convert back to defaultdict

        return conjuncted_concepts

    @staticmethod
    def ensemble_predictions(baseline_matched_concepts, dependency_matched_concepts):
        ensembled_matched_concepts = {}
        for concept, annotations in baseline_matched_concepts.items():
            if annotations:
                ensembled_annotations = dependency_matched_concepts[concept]
            else:
                ensembled_annotations = []
            ensembled_matched_concepts[concept] = ensembled_annotations

        # preserve data format of baseline_matched_concepts and dependency_matched_concepts for confusion matrix
        ensembled_matched_concepts = defaultdict(lambda: defaultdict(list), ensembled_matched_concepts)

        return ensembled_matched_concepts

    def baseline_detector_premodification(self, valid_modality_concept_matches, concept_tagsets, metadata):
        matched_concepts = defaultdict(lambda: defaultdict(list))
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            start_modality_index = cue_idxs[0]

            lexical_modality_cue = metadata['negation'][modality_cue]['cue']
            # assign to each modality cue the nearest concept

            # check for pre-modifier modality cues
            if lexical_modality_cue in self.negation_cues:
                # extract first following concept
                concept_start_idxs = {concept: min(tagset) for
                                      concept, tagset in concept_tagsets.items()
                                      if min(tagset) > start_modality_index}
                if concept_start_idxs:
                    first_following_concept = sorted(concept_start_idxs.items(), key=lambda x:x[1])[0][0]
                    # match with first following concept if allowed
                    if first_following_concept in valid_modality_concept_matches[(modality_cue, cue_idxs)]:
                        matched_concepts[first_following_concept]['pre'].append(lexical_modality_cue)

        return matched_concepts

    def dependency_detector(self, valid_modality_concept_matches, concept_tagsets, metadata, depdata):
        matched_concepts = defaultdict(list)
        dependency_tree = self.build_dependency_tree(depdata)
        for (modality_cue, cue_idxs), valid_concepts in valid_modality_concept_matches.items():
            lexical_modality_cue = metadata['negation'][modality_cue]['cue']
            # assign to each modality cue all concepts within a dependency span
            # pre- or post-modifiers become irrelevant here, should be inferred by the dependency parse...

            # traverse tree, assign all concepts within n dependencies
            # accumulate paths of all token idxs of the cue
            for cue_idx in cue_idxs:
                dependent_terms, governor_terms = self.extract_dependency_path(dependency_tree, cue_idx)
                idxs_to_match = set()  # extract from dependent terms
                for numsteps, dep_idxs in dependent_terms.items():
                    if numsteps <= self.allowed_dependency_steps:
                        idxs_to_match.update(dep_idxs)
                for concept, concept_idxs in concept_tagsets.items():
                    if concept in valid_concepts:
                        if idxs_to_match.intersection(concept_idxs):
                            matched_concepts[concept].append(lexical_modality_cue)

        return matched_concepts

    def extract_dependency_path(self, dependency_tree, token_index):
        forward_dependency = dependency_tree['forward']
        backward_dependency = dependency_tree['backward']
        dependent_terms = defaultdict(list)
        governor_terms = defaultdict(list)
        steps = 1
        self.traverse_dependency_path(token_index, forward_dependency, dependent_terms, steps)
        self.traverse_dependency_path(token_index, backward_dependency, governor_terms, steps)

        return dependent_terms, governor_terms

    def traverse_dependency_path(self, reference_idx, dependency_tree, token_idxs, steps):
        dependencies = dependency_tree[reference_idx]
        token_idxs[steps] += dependencies
        steps += 1
        for dependency in dependencies:
            self.traverse_dependency_path(dependency, dependency_tree, token_idxs, steps)

    @staticmethod
    def extract_concept_tagsets(tags):
        concepts_idxs = defaultdict(set)
        for i, position_tags in enumerate(tags):
            for tag in position_tags:
                if tag.startswith('C'):
                    concepts_idxs[tag].add(i)

        return concepts_idxs

    @staticmethod
    def extract_valid_modality_concept_matches(tags):
        # disables modality cues for concepts which they are a part of!
        modality_tagsets = defaultdict(set)
        concept_tagsets = defaultdict(set)
        for i, position_tags in enumerate(tags):
            for tag in position_tags:
                if tag.startswith('negation'):
                    modality_tagsets[tag].add(i)
                elif tag.startswith('C'):
                    concept_tagsets[tag].add(i)
                else:
                    pass

        valid_modality_concept_matches = defaultdict(list)
        for modality_cue, cue_tagindexset in modality_tagsets.items():
            for concept, concept_tagindexset in concept_tagsets.items():
                if not cue_tagindexset.intersection(concept_tagindexset):
                    modality_data = (modality_cue, tuple(sorted(cue_tagindexset)))
                    valid_modality_concept_matches[modality_data].append(concept)

        return valid_modality_concept_matches

    @staticmethod
    def print_dependency_tree(dependency_tree, tokens):
        for dep, heads in dependency_tree['forward'].items():
            for head in heads:
                print(tokens[dep], '--->', tokens[head])

    @staticmethod
    def build_dependency_tree(depdata):
        # build dependency dict
        forward_dependency = defaultdict(set)
        # tokens, depdata = list(zip(*sentence))
        for index, (dep, depindex) in enumerate(depdata):
            depindex = int(depindex)
            if depindex > 0:  # don't include ROOT for dependency
                real_depindex = depindex - 1
                forward_dependency[index].add(real_depindex)

        # create backward dependency
        backward_dependency = defaultdict(set)
        for dependent_index, governor_indexes in forward_dependency.items():
            for governor_index in governor_indexes:
                backward_dependency[governor_index].add(dependent_index)

        dependency_tree = {'forward': forward_dependency, 'backward': backward_dependency}

        return dependency_tree


class NegationDetectorEvaluation(NegationDetector):

    def __init__(self):

        super(NegationDetectorEvaluation, self).__init__()

        self.evaluation_mode = True
        self.confusion_matrix = {'true_pos': [],
                                 'true_neg': [],
                                 'false_pos': [],
                                 'false_neg': []}

    def __call__(self, sentence_instances, model, outfile='', verbose=True):

        self.reset_confusion_matrix()
        for sentence_instance in sentence_instances:
            self.detect_and_evaluate_instance(sentence_instance, model)

        results = self.evaluation()
        if verbose:
            print(results)

        if outfile:
            results_outfile = '{}_results.json'.format(outfile)
            confusion_matrix_outfile = '{}_confusion.json'.format(outfile)
            print('Saving...')
            with open(outfile, 'w') as f:
                json.dump(results_outfile, f)
            print('Done')
            self.save_confusion_matrix(confusion_matrix_outfile)

        return results

    def detect_and_evaluate_instance(self, sentence_instance, model):
        self.assert_gold_data(sentence_instance)
        detection_data = self.detect(sentence_instance, model)
        self.update_confusion_matrix(*detection_data)

    @staticmethod
    def assert_gold_data(instance):
        if 'true_modality' not in instance[0][1]['concepts']['C0']:
            raise ValueError('This is not proper gold data, please provide gold data')

    def reset_confusion_matrix(self):
        self.confusion_matrix = {'true_pos': [], 'true_neg': [], 'false_pos': [], 'false_neg': []}

    def save_confusion_matrix(self, outfile):
        print('Saving confusion matrix...')
        with open(outfile, 'w') as f:
            json.dump(self.confusion_matrix, f)
        print('Done')

    def update_confusion_matrix(self, matched_concepts, concepts, metadata, instance_id):
        for concept in concepts:
            if matched_concepts[concept]:
                modality_prediction = matched_concepts[concept]
                # check for true positive
                if metadata['concepts'][concept]['true_modality'] == 'negation':
                    self.confusion_matrix['true_pos'].append((concept, instance_id, modality_prediction))
                # check for false positive
                else:
                    self.confusion_matrix['false_pos'].append((concept, instance_id, modality_prediction))
            else:
                # check for false negative
                if metadata['concepts'][concept]['true_modality'] == 'negation':
                    self.confusion_matrix['false_neg'].append((concept, instance_id))
                # check for true negative
                else:
                    self.confusion_matrix['true_neg'].append((concept, instance_id))

    def evaluation(self):
        # uses the confusion matrix to calculate various evaluation metrics

        true_pos = len(self.confusion_matrix['true_pos'])
        true_neg = len(self.confusion_matrix['true_neg'])
        false_pos = len(self.confusion_matrix['false_pos'])
        false_neg = len(self.confusion_matrix['false_neg'])

        try:
            positive_precision = true_pos / (true_pos + false_pos)
        except ZeroDivisionError:
            positive_precision = None
        try:
            positive_recall = true_pos / (true_pos + false_neg)
        except ZeroDivisionError:
            positive_recall = None
        try:
            negative_precision = true_neg / (true_neg + false_neg)
        except ZeroDivisionError:
            negative_precision = None
        try:
            negative_recall = true_neg / (true_neg + false_pos)
        except ZeroDivisionError:
            negative_recall = None

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        majority_baseline = max(true_pos + false_neg, true_neg + false_pos) / (true_pos + false_neg + true_neg + false_pos)

        results = {'accuracy': accuracy,
                   'majority_baseline': majority_baseline,
                   'positive_precision': positive_precision,
                   'positive_recall': positive_recall,
                   'negative_precision': negative_precision,
                   'negative_recall': negative_recall}

        return results

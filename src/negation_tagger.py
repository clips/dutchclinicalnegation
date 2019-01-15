from collections import defaultdict
from collections import namedtuple
import json
from negation_triggers import negation_triggers


class NegationTagger(object):

    def __init__(self):

        self.negation_cues = negation_triggers['pre']
        self.gold = False

        self.Concept_instance = namedtuple('Concept_instance',
                                           ('Sentence', 'Concept', 'Concept_span', 'Sentence_id', 'File_id')
                                           )
        self.Concept_instance_gold = namedtuple('Concept_instance_gold',
                                           ('Sentence', 'Concept', 'Modality', 'Concept_span', 'Sentence_id', 'File_id')
                                           )

    def __call__(self, concept_instances, outfile=''):

        self.assert_gold_status(concept_instances[0])
        tagged_sentences = []
        tagged_sentences_ids = []
        grouped_sentences = self.group_sentences(concept_instances)
        for file_id in grouped_sentences:
            for sentence_id in grouped_sentences[file_id]:
                sentence_concept_instances = grouped_sentences[file_id][sentence_id]
                tagged_sentence = self.tag_fused_sentence(sentence_concept_instances)
                tagged_sentences.append(tagged_sentence)
                tagged_sentences_ids.append((file_id, sentence_id))

        sentence_instances = list(zip(tagged_sentences, tagged_sentences_ids))

        data = {'sentence_instances': sentence_instances, 'gold': self.gold}

        if outfile:
            print('Saving...')
            with open('{}'.format(outfile), 'w') as f:
                json.dump(data, f)
            print('Done')

        return data

    def load_concept_instances(self, infile):
        # loads concept instances
        with open(infile, 'r') as f:
            concept_instances = json.load(f)

        # convert to proper format
        concept_instances = self.convert_concept_instances(concept_instances)

        return concept_instances

    def convert_concept_instances(self, concept_instances):

        if len(concept_instances[0]) == 6:
            concept_instances = [self.Concept_instance_gold(*x) for x in concept_instances['instances']]
        elif len(concept_instances[0]) == 5:
            concept_instances = [self.Concept_instance(*x) for x in concept_instances['instances']]
        else:
            raise ValueError('Data is invalid, fits neither of the data structures')

        return concept_instances

    def assert_gold_status(self, instance):

        data_type = type(instance).__name__
        if data_type == 'Concept_instance':
            self.gold = False
        elif data_type == 'Concept_instance_gold':
            self.gold = True
        else:
            raise ValueError('Concept instance should be namedtuple class attribute, not {}'.format(data_type))

    @staticmethod
    def group_sentences(concept_instances):
        # collect data of sentences
        grouped_sentences = defaultdict(lambda: defaultdict(list))
        # group all concepts per sentence, fuse those with same sentence
        for concept_instance in concept_instances:
            file_id = concept_instance.File_id
            sentence_id = concept_instance.Sentence_id
            grouped_sentences[file_id][sentence_id].append(concept_instance)

        return grouped_sentences

    def tag_fused_sentence(self, sentence_instances):
        sentence = sentence_instances[0].Sentence
        sentence_tags = [[] for _ in range(len(sentence))]
        metadata = defaultdict(lambda: defaultdict(dict))

        # tag modality
        sentence_tags, metadata = self.tag_modality_cues(sentence_tags, metadata, sentence)

        # tag concepts
        for instance in sentence_instances:
            sentence_tags, metadata = self.tag_concept(sentence_tags, metadata, instance)

        return sentence_tags, dict(metadata), sentence

    def tag_concept(self, sentence_tags, metadata, instance):
        sentence = instance.Sentence
        tokens, spans, _ = list(zip(*sentence))

        # tag concept
        concept_span = instance.Concept_span
        concept_start_position, concept_end_position = None, None
        for i, (begin_idx, end_idx) in enumerate(spans):
            if begin_idx <= concept_span[0] < end_idx:
                concept_start_position = i
            if begin_idx < concept_span[1] <= end_idx:
                concept_end_position = i
        assert (concept_start_position is not None) and (concept_end_position is not None), \
            'No concept boundaries found for {}'.format(instance)

        concept_tag = 'C{}'.format(len(metadata['concepts']))

        if self.gold:
            metadata['concepts'][concept_tag] = {'true_modality': instance.Modality,
                                                 'concept': instance.Concept}
        else:
            metadata['concepts'][concept_tag] = {'concept': instance.Concept}

        for i in range(concept_start_position, concept_end_position + 1):
            sentence_tags[i].append(concept_tag)

        return sentence_tags, metadata

    def tag_modality_cues(self, sentence_tags, metadata, sentence):
        tokens, _, _ = list(zip(*sentence))
        tokens = [token.lower() for token in tokens]

        # track all matching modality idxs
        modality_triggers = self.negation_cues
        sorted_modality_cues = sorted(modality_triggers, key=lambda x: len(x.split()), reverse=True)
        modality_matches = []
        for modality_cue in sorted_modality_cues:
            sequence_to_match = modality_cue.split()
            matching_idxs = self.find_subsequence(sequence_to_match, tokens)
            for match_idxs in matching_idxs:
                modality_matches.append((modality_cue, match_idxs))

        # make sure that no token is annotated with more than one negation instance
        # we already ordered the modality cues from long to short, so encompassed modality cues will be ignored
        unique_modality_matches = []
        matched_idxs = set()
        for (modality_cue, match_idxs) in modality_matches:
            if not set(match_idxs).intersection(matched_idxs):
                unique_modality_matches.append((modality_cue, match_idxs))
                matched_idxs.update(match_idxs)

        # create metalabels
        modality_matches = sorted(unique_modality_matches, key=lambda x:x[1][0])
        for (modality_cue, match_idxs) in modality_matches:
            modality_tag = 'negation_{}'.format(len(metadata))
            metadata['negation'][modality_tag] = {'cue': modality_cue}
            for i in match_idxs:
                sentence_tags[i].append(modality_tag)

        return sentence_tags, metadata

    @staticmethod
    def find_subsequence(subsequence, sequence):
        # matches a subsequence of tokens in a token sequence
        match_idxs = []
        subsequence_len = len(subsequence)
        start_token = subsequence[0]
        start_idxs = [i for i, item in enumerate(sequence) if item == start_token]
        if not start_idxs:
            return match_idxs

        end_ranges = [i + subsequence_len for i in start_idxs]
        for start_idx, end_range in zip(start_idxs, end_ranges):
            if sequence[start_idx:end_range] == subsequence:
                sub_idxs = list(range(start_idx, end_range))
                match_idxs.append(sub_idxs)

        return match_idxs

import json
import re
from collections import namedtuple
import subprocess


class PreprocessCorpus:

    def __init__(self):

        self.frog_labels = {0: 'index',
                            1: 'text',
                            2: 'lemma',
                            3: 'morph',
                            4: 'pos',
                            5: 'posprob',
                            6: 'ner',
                            7: 'chunker',
                            8: 'depindex',
                            9: 'dep'}

        self.Concept_instance = namedtuple('Concept_instance',
                                           ('Sentence', 'Concept', 'Concept_span', 'Sentence_id', 'File_id')
                                           )

        self.Concept_instance_gold = namedtuple('Concept_instance_gold',
                                           ('Sentence', 'Concept', 'Modality', 'Concept_span', 'Sentence_id', 'File_id')
                                           )

    def __call__(self, file_ids, outfile=''):

        accumulated_concept_instances = []
        for file_id in file_ids:
            accumulated_concept_instances += self.preprocess_file(file_id)

        if outfile:
            print('Saving...')
            with open('{}'.format(outfile), 'w') as f:
                json.dump(accumulated_concept_instances, f)
            print('Done')

        return accumulated_concept_instances

    def preprocess_file(self, file_id, already_frogged=False, corpusfile=False, outfile=''):

        with open(file_id, 'r') as f:
            data = json.load(f)

        if corpusfile:
            data = self.convert_corpusfile_to_general_format(data)

        if already_frogged:
            frog_file = '{}_frog'.format(file_id)
            with open(frog_file, 'r') as f:
                frogged_text = json.load(f)
            if corpusfile:
                frogged_text = self.convert_corpus_frog_output(frogged_text)
        else:
            frogged_text = self.frog_file(file_id)

        concept_instances = self.extract_instances(data, frogged_text, file_id)

        if outfile:
            print('Saving...')
            with open('{}'.format(outfile), 'w') as f:
                json.dump(concept_instances, f)
            print('Done')

        return concept_instances

    def frog_file(self, file_id):
        with open(file_id, 'r') as f:
            data = json.load(f)

        text = data['text']
        path_to_text = '{}_text'.format(file_id)
        with open(path_to_text, 'w') as f:
            f.write(text)
        frogged_text = subprocess.check_output(['frog', path_to_text])
        frogged_text = frogged_text.decode()
        frogged_text = self.convert_raw_frog_output(frogged_text)

        frog_file = '{}_frog'.format(file_id)
        with open(frog_file, 'w') as f:
            json.dump(frogged_text, f)

        return frogged_text

    @staticmethod
    def convert_corpusfile_to_general_format(data):
        general_format = dict()
        general_format['text'] = data['text']
        general_format['concept_spans'] = []
        general_format['negation_status'] = []

        modality = data['modifications']
        negated_concepts = [x['obj'] for x in modality if x['pred'] == 'Negation']

        for concept_dict in data['denotations']:
            span = concept_dict['span']
            if concept_dict['id'] in negated_concepts:
                negation_status = True
            else:
                negation_status = False
            general_format['concept_spans'].append(span)
            general_format['negation_status'].append(negation_status)

        return general_format

    def extract_instances(self, data, frogged_text, file_id):
        # collect data
        text = data['text']
        tokenized_sentences = self.frogged_sentences(text, frogged_text)

        # post-processing: collect annotations to readjust sentence boundaries to accommodate for concept spans
        concept_spans = [(span['begin'], span['end']) for span in data['concept_spans']]
        tokenized_sentences = self.match_sentence_splitting_with_concepts(concept_spans, tokenized_sentences)

        concept_instances = []
        for i, concept_span in enumerate(concept_spans):
            concept = text[concept_span[0]:concept_span[1]]
            sentence, sentence_id = self.find_sentence(concept_span, tokenized_sentences)
            if not sentence:
                print("No sentence found for concept '{}' ".format(concept))
                continue
            if 'negation_status' in data:
                modality = data['negation_status'][i]
                concept_instance = self.Concept_instance_gold(
                    sentence, concept, modality, concept_span, sentence_id, file_id)
            else:
                concept_instance = self.Concept_instance(
                    sentence, concept, concept_span, sentence_id, file_id)
            concept_instances.append(concept_instance)

        return concept_instances

    def convert_raw_frog_output(self, frogged_output):
        frogged_sentences = frogged_output.split('\n\n')
        converted_sentences = []
        for sentence in frogged_sentences:
            if sentence:
                current_sentence = []
                tokens_data = sentence.split('\n')
                for token_data in tokens_data:
                    fields = token_data.split('\t')
                    token_dict = {self.frog_labels[i]: field for i, field in enumerate(fields)}
                    # extract the specific information to be used
                    relevant_token_data = (token_dict['text'], (token_dict['dep'], token_dict['depindex']))
                    current_sentence.append(relevant_token_data)
                converted_sentences.append(current_sentence)

        return converted_sentences

    def convert_corpus_frog_output(self, frogged_output):
        converted_sentences = []
        current_sentence = []
        for token_dict in frogged_output:
            relevant_token_data = (token_dict['text'], (token_dict['dep'], token_dict['depindex']))
            current_sentence.append(relevant_token_data)
            if 'eos' in token_dict:
                converted_sentences.append(current_sentence)
                current_sentence = []

        return converted_sentences

    @staticmethod
    def frogged_sentences(text, tokenized_sentences):
        # map tokens from Frog output to raw text and assign the relevant frog labels
        text_tokens = []
        current_text = text
        current_index = 0
        for sentence in tokenized_sentences:
            tokens_sentence = []
            for token, relevant_token_data in sentence:
                # an empty token means something went wrong
                if not token:
                    raise ValueError("Empty token in sentence '{}'".format(sentence))
                # convert _-padding back to whitespace
                if '_' in token:
                    converted_token = ''.join([re.escape(t) + r'(\s|\n|_)*' for t in token.split('_')])
                else:
                    converted_token = re.escape(token)
                token_matcher = re.compile(converted_token)
                # if a token cannot be matched in the text, something went wrong...
                try:
                    span = token_matcher.search(current_text).span()
                except AttributeError:
                    raise ValueError("Something went wrong, tried to match '{}' in text '{}'".format(
                        token, current_text))
                # we track the indices of the token by caching the accumulated start index and updating it continuously
                begin_index, end_index = span
                begin_index += current_index
                end_index += current_index
                token_data = (token, (begin_index, end_index), relevant_token_data)
                tokens_sentence.append(token_data)
                current_index = end_index
                current_text = text[current_index:]
                text_tokens.append(tokens_sentence)

        return text_tokens

    @staticmethod
    def find_sentence(span, tokenized_sentences):
        for id, sentence in enumerate(tokenized_sentences):
            first_token_index = sentence[0][1][0]
            last_token_index = sentence[-1][1][1]
            begin_index, end_index = span
            if first_token_index <= begin_index:
                if end_index <= last_token_index:
                    sentence_id = 'S{}'.format(id)
                    return sentence, sentence_id

    @staticmethod
    def match_sentence_splitting_with_concepts(spans, tokenized_sentences):
        redistributed_sentences = []
        tokenized_sentences = iter(tokenized_sentences)
        current_sentence = []
        while True:
            try:
                sentence = next(tokenized_sentences)
            except StopIteration:
                return redistributed_sentences
            current_sentence += sentence
            sentence_boundary = sentence[-1][1][1]
            greenlight = True
            for span in spans:
                begin_index, end_index = span
                if begin_index < sentence_boundary < end_index:
                    greenlight = False
                    break
            if greenlight:
                redistributed_sentences.append(current_sentence)
                current_sentence = []

        raise ValueError('Process did not end successfully...')

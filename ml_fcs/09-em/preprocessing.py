from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r') as file:
        # Заменяем '&' на '&amp, потому что этот значок все портил при чтении некоторых файлов
        xml_content = file.read().replace("&", "&amp;")

    root = ET.fromstring(xml_content)
    sentences = []
    labels = []
    for child in root:
        array = []
        for c in child:
            array.append(c.text if c.text is not None else '')

        sentences.append(SentencePair(array[0].split(), array[1].split()))
        list_of_tuples_eng = [tuple(map(int, pair.split('-'))) for pair in array[2].split()]
        list_of_tuples_che = [tuple(map(int, pair.split('-'))) for pair in array[3].split()]
        labels.append(LabeledAlignment(list_of_tuples_eng, list_of_tuples_che))

    return (sentences, labels)


pass


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    keys_eng = set()
    keys_che = set()

    for sentence_pair in sentence_pairs:
        keys_eng = keys_eng | (set(sentence_pair.source))
        keys_che = keys_che | (set(sentence_pair.target))

    values_eng = [i for i in range(len(keys_eng))]
    values_che = [i for i in range(len(keys_che))]

    dict_eng = dict(zip(keys_eng, values_eng))
    dict_che = dict(zip(keys_che, values_che))

    return dict_eng, dict_che


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for sentence_pair in sentence_pairs:
        eng_encoded_sentence = np.array([source_dict[word] for word in sentence_pair.source])
        che_encoded_sentence = np.array([target_dict[word] for word in sentence_pair.target])

        tokenized_sentence_pairs.append(TokenizedSentencePair(eng_encoded_sentence, che_encoded_sentence))

    return tokenized_sentence_pairs

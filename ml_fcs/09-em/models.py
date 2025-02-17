from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple
from scipy.special import softmax, xlogy

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        posteriors = []
        for sentence_pair in parallel_corpus:
            source = sentence_pair.source_tokens.reshape(-1, 1)
            target = sentence_pair.target_tokens.reshape(1, -1)

            post = self.translation_probs[source, target] / np.sum(self.translation_probs[source, target], axis=0)

            posteriors.append(post)
        return posteriors

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        elbos = 0
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            source = sentence_pair.source_tokens.reshape(-1, 1)
            target = sentence_pair.target_tokens.reshape(1, -1)

            elbos += (posterior * np.log(self.translation_probs[source, target])).sum() - xlogy(posterior,
                                                                                                posterior.shape[
                                                                                                    0] * posterior).sum()
        return elbos

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs -= self.translation_probs
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):

            for i in enumerate(sentence_pair.source_tokens):
                for j in enumerate(sentence_pair.target_tokens):
                    self.translation_probs[i[1], j[1]] += posterior[i[0], j[0]]

        self.translation_probs /= self.translation_probs.sum(axis=1).reshape(self.num_source_words, -1)

        return self._compute_elbo(parallel_corpus, posteriors)
        pass

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)

            elbo = self._m_step(parallel_corpus, posteriors)

            history.append(elbo)

        return history

    def align(self, sentences):
        result = []
        posteriors = self._e_step(sentences)
        for sentence, posterior in zip(sentences, posteriors):
            alignment = []
            arr = np.argmax(posterior, axis=0) + 1
            for i in range(arr.shape[0]):
                alignment.append((arr[i], i + 1))
            result.append(alignment)

        return result


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        key = (src_length, tgt_length)
        if key not in self.alignment_probs:
            self.alignment_probs[key] = np.full(key, 1 / src_length, dtype=np.float32)
        elif (self.alignment_probs[key].sum() != key[1]):
            self.alignment_probs[key] /= self.alignment_probs[key].sum(axis=0)

        return self.alignment_probs[(src_length, tgt_length)]

    def _e_step(self, parallel_corpus):
        posteriors = []
        for sentence_pair in parallel_corpus:
            source = sentence_pair.source_tokens.reshape(-1, 1)
            target = sentence_pair.target_tokens.reshape(1, -1)
            phi = self._get_probs_for_lengths(source.shape[0], target.shape[1])
            post = (self.translation_probs[source, target] * phi) / np.sum(self.translation_probs[source, target] * phi,
                                                                           axis=0)

            posteriors.append(post)

        return posteriors
        pass

    def _compute_elbo(self, parallel_corpus, posteriors):
        elbos = 0
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            source = sentence_pair.source_tokens.reshape(-1, 1)
            target = sentence_pair.target_tokens.reshape(1, -1)
            phi = self._get_probs_for_lengths(source.shape[0], target.shape[1])

            elbos += (posterior * np.log(self.translation_probs[source, target])).sum() + xlogy(posterior,
                                                                                                phi / posterior).sum()
        return elbos
        pass

    def _m_step(self, parallel_corpus, posteriors):
        self.translation_probs -= self.translation_probs
        self.alignment_probs.clear()
        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            source = sentence_pair.source_tokens
            target = sentence_pair.target_tokens
            key = (len(source), len(target))
            if key not in self.alignment_probs:
                self.alignment_probs[key] = np.zeros(key)
            self.alignment_probs[key] += posterior
            for i in enumerate(sentence_pair.source_tokens):
                for j in enumerate(sentence_pair.target_tokens):
                    self.translation_probs[i[1], j[1]] += posterior[i[0], j[0]]

        self.translation_probs /= self.translation_probs.sum(axis=1).reshape(self.num_source_words, -1)
        return self._compute_elbo(parallel_corpus, posteriors)
        pass

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)

            elbo = self._m_step(parallel_corpus, posteriors)

            history.append(elbo)

        return history

    def align(self, sentences):
        result = []
        posteriors = self._e_step(sentences)
        for sentence, posterior in zip(sentences, posteriors):
            alignment = []
            arr = np.argmax(posterior, axis=0) + 1
            for i in range(arr.shape[0]):
                alignment.append((arr[i], i + 1))
            result.append(alignment)

        return result

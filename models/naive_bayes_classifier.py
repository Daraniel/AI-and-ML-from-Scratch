import codecs
import math
import os
import re
from typing import Dict, List, Tuple, Union

from common.base_model import Classifier


class NaiveBayesClassifier(Classifier):
    def __init__(self, min_count=1):
        """
        this classifier classifies the text using Naive Bayes method. it takes text as input.
        :param min_count: minimum iterations counts for each word to consider it a valid word in the vocabulary
        """
        self.min_count = min_count
        self.vocabulary = {}
        self.doc_groups = {}
        self.priors = {}
        self.conditionals = {}

    @staticmethod
    def _tokenize(string: str):
        """
        tokenizes a string and returns words with length at least two
        :param string: string to tokenize
        :return: tokens of the string
        """
        return re.findall(r"\b\w\w+\b", string)

    @staticmethod
    def _tokenize_and_parse_doc(
        doc: str,
        content: Dict[str, Union[int, Dict[str, int]]],
        vocabulary: Dict[str, int],
    ):
        """
        tokenizes and parses a document string and updates the content and the vocabulary with it
        :param doc: document to check
        :param content: dictionary containing the content
        :param vocabulary: dictionary of vocabulary
        """
        terms = NaiveBayesClassifier._tokenize(doc)
        content["term_count"] += len(terms)
        for term in terms:
            if term in content["terms"]:
                vocabulary[term] += 1
                content["terms"][term] += 1
            else:
                content["terms"][term] = 1
                if term in vocabulary:
                    vocabulary[term] += 1
                else:
                    vocabulary[term] = 1

    def learn(self, doc_groups: Dict[str, List[str]]):
        """
        trains the Naive Bayes model on the input doc_groups
        :param doc_groups: dictionary that has the document groups as the key and list of the content of documents of
        that group as value
        """
        parsed_doc_groups = {}
        vocabulary = {}

        for doc_group, docs in doc_groups.items():
            parsed_doc_groups[doc_group] = {
                "document_count": len(docs),
                "term_count": 0,
                "terms": dict(),
            }
            for doc in docs:
                self._tokenize_and_parse_doc(
                    doc, parsed_doc_groups[doc_group], vocabulary
                )

        return self._learn(parsed_doc_groups, vocabulary)

    def _learn(
        self,
        doc_groups: Dict[str, Dict[str, Union[int, Dict[str, int]]]],
        vocabulary: Dict[str, int],
    ):
        """
        trains the Naive Bayes model on the input doc_groups and with given vocabulary
        :param doc_groups: doc_groups to train the model on
        :param vocabulary: words that happened in the vocabulary
        """
        self.doc_groups = doc_groups
        self.vocabulary = {
            term: term_count
            for term, term_count in vocabulary.items()
            if term_count > self.min_count
        }

        num_docs = sum([doc["document_count"] for doc in self.doc_groups.values()])

        for doc_group in self.doc_groups:
            self.priors[doc_group] = math.log(
                self.doc_groups[doc_group]["document_count"] / num_docs
            )

            self.conditionals[doc_group] = {}
            terms_in_class = sum(self.doc_groups[doc_group]["terms"].values())

            for term in self.vocabulary:
                self.conditionals[doc_group][term] = math.log(
                    (self.doc_groups[doc_group]["terms"].get(term, 0) + 1)
                    / (terms_in_class + len(self.vocabulary))
                )

    def infer(self, document: str) -> Tuple[Dict[str, float], int]:
        """
        tokenizes and classifies given document
        :param document: document to classify
        :return: confidence scores of the document and its class
        """
        tokens = self._tokenize(document)
        scores = {}
        for doc_group in self.doc_groups:
            scores[doc_group] = self.priors[doc_group]
            for term in tokens:
                scores[doc_group] += self.conditionals[doc_group].get(term, 0)
        return scores, max(scores, key=scores.get)


class NaiveBayesClassifierExtended(NaiveBayesClassifier):
    def __init__(self, min_count=1):
        """
        this classifier extends the NaiveBayesClassifier and allows it to work with documents in a given path
        :param min_count: minimum iterations counts for each word to consider it a valid word in the vocabulary
        """
        super().__init__(min_count)

    @staticmethod
    def _read_file(doc_path: str) -> str:
        """
        this method reads the file in the given path and removes its header and returns its text
        :param doc_path: path of the document to read
        :return: body of the document
        """
        with codecs.open(doc_path, encoding="latin1") as doc:
            doc = doc.read().lower()
            _, _, body = doc.partition("\n\n")
            return body

    def _parse_path(
        self, path: str
    ) -> Tuple[Dict[str, Dict[str, Union[int, Dict[str, int]]]], Dict[str, int]]:
        """
        extracts all documents in different classes in a given path and creates their respective document groups and
        extracts their vocabulary
        :param path: path to parse documents from
        :return: document groups and vocabulary
        """
        self.num_docs = 0
        doc_groups = {}
        vocabulary = {}
        for doc_group in os.listdir(path):
            docs_path = os.path.join(path, doc_group)
            docs = os.listdir(docs_path)
            doc_groups[doc_group] = {
                "document_count": len(docs),
                "term_count": 0,
                "terms": dict(),
            }

            for doc_path in docs:
                doc = self._read_file(os.path.join(docs_path, doc_path))
                self._tokenize_and_parse_doc(doc, doc_groups[doc_group], vocabulary)
        return doc_groups, vocabulary

    def train(self, path: str):
        """
        trains the Naive Bayes model on the documents in the given path
        :param path: path of the documents. it must contain documents in folders that determine their classes
        """
        self._learn(*self._parse_path(path))

    def test(self, path: str) -> Tuple[List[int], List[int]]:
        """
        tests the performance of the model on the documents in the given path
        :param path: path of the documents. it must contain documents in folders that determine their classes
        :return: true and predicted classes of the documents
        """
        true_y = []
        pred_y = []

        for c in self.doc_groups:
            for f in os.listdir(os.path.join(path, c)):
                doc_path = os.path.join(path, c, f)
                _, result_class = self.infer(self._read_file(doc_path))
                pred_y.append(result_class)
                true_y.append(c)
        return true_y, pred_y

    def predict(self, path: str) -> Tuple[Dict[str, float], int]:
        """
        predicts the class of the given document
        :param path: path of the document to predict
        :return: confidence scores of the document and its class
        """
        return self.infer(self._read_file(path))

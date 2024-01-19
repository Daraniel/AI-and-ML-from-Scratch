from models.naive_bayes_classifier import NaiveBayesClassifier


class TestNaiveBayesClassifier:
    def setup_method(self):
        self.classifier = NaiveBayesClassifier()

    def test_learn(self):
        doc_groups = {
            "group1": ["This is a test document.", "This is another test document."],
            "group2": ["This is a test document.", "This is another test document."],
        }
        self.classifier.learn(doc_groups)
        assert len(self.classifier.doc_groups) == 2
        assert len(self.classifier.vocabulary) > 0
        assert len(self.classifier.priors) == 2
        assert len(self.classifier.conditionals) == 2

    def test_infer(self):
        doc_groups = {
            "group1": ["This is a test document.", "This is another test document."],
            "group2": ["This is a test document.", "This is another test document."],
        }
        doc = "This is a test document."
        self.classifier.learn(doc_groups)
        scores, predicted_class = self.classifier.infer(doc)
        assert len(scores) == 2
        assert predicted_class in scores

    def test_tokenize(self):
        text = "This is a test document."
        tokens = self.classifier._tokenize(text)
        assert len(tokens) == 4
        assert "test" in tokens
        assert "document" in tokens

    def test_tokenize_and_parse_doc(self):
        doc = "This is a test document."
        content = {"document_count": 1, "term_count": 0, "terms": {}}
        vocabulary = {"test": 0, "document": 0}
        self.classifier._tokenize_and_parse_doc(doc, content, vocabulary)
        assert content["term_count"] == 4
        assert "test" in content["terms"]
        assert "document" in content["terms"]
        assert vocabulary["test"] == 1
        assert vocabulary["document"] == 1

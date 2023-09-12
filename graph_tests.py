import networkx as nx
import numpy as np
import unittest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from graph import (
    create_graph_from_count_matrix, create_graph_from_tfidf_matrix,
    create_character_graph, create_stopword_punctuation_graph,
    convert_graph_to_numpy, convert_graphs_to_numpy, convert_numpy_to_umap
)
from signature import calculate_character_frequencies, calculate_stopword_and_nonletter_frequencies, normalize_text


class GraphTests(unittest.TestCase):

    def setUp(self):
        self.text = "This is a sample text for graph creation."
        self.normalized_text = normalize_text(self.text)

    def test_create_graph_from_count_matrix(self):
        count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform([self.normalized_text])
        graph = create_graph_from_count_matrix(count_matrix)
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))

    def test_create_graph_from_tfidf_matrix(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([self.normalized_text])
        graph = create_graph_from_tfidf_matrix(tfidf_matrix)
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))

    def test_create_character_graph(self):
        graph = create_character_graph(self.normalized_text)
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))

    def test_create_stopword_punctuation_graph(self):
        graph = create_stopword_punctuation_graph(self.normalized_text)
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))

    def test_convert_graph_to_numpy(self):
        graph = create_graph_from_count_matrix(CountVectorizer().fit_transform([self.normalized_text]))
        adj_matrix, node_indices = convert_graph_to_numpy(graph)
        self.assertIsInstance(adj_matrix, np.ndarray)
        self.assertIsInstance(node_indices, dict)

    def test_convert_graphs_to_numpy(self):
        bag_of_words_graph = create_graph_from_count_matrix(CountVectorizer().fit_transform([self.normalized_text]))
        tfidf_graph = create_graph_from_tfidf_matrix(TfidfVectorizer().fit_transform([self.normalized_text]))
        graphs = {
            'bag_of_words_graph': bag_of_words_graph,
            'tfidf_graph': tfidf_graph
        }
        numpy_graphs = convert_graphs_to_numpy(graphs)
        self.assertEqual(len(numpy_graphs), 2)
        self.assertIsInstance(numpy_graphs['bag_of_words_graph'][0], np.ndarray)
        self.assertIsInstance(numpy_graphs['bag_of_words_graph'][1], dict)
        self.assertIsInstance(numpy_graphs['tfidf_graph'][0], np.ndarray)
        self.assertIsInstance(numpy_graphs['tfidf_graph'][1], dict)

    def test_convert_numpy_to_umap(self):
        graph = create_graph_from_count_matrix(CountVectorizer().fit_transform([self.normalized_text]))
        numpy_graph = convert_graph_to_numpy(graph)
        umap_embedding = convert_numpy_to_umap(numpy_graph)
        self.assertIsInstance(umap_embedding, np.ndarray)
        self.assertEqual(umap_embedding.shape[1], 2)


if __name__ == '__main__':
    unittest.main()

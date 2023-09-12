import unittest
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from graph import create_graph_from_count_matrix, create_graph_from_tfidf_matrix, create_character_graph, create_word_graph, create_stopword_punctuation_graph, convert_graph_to_numpy, convert_graphs_to_numpy, convert_numpy_to_umap


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.text = "This is a test text."

    def test_create_graph_from_count_matrix(self):
        count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform([self.text])
        graph = create_graph_from_count_matrix(count_matrix)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge('This', 'is', weight=1)
        expected_graph.add_edge('is', 'a', weight=1)
        expected_graph.add_edge('a', 'test', weight=1)
        expected_graph.add_edge('test', 'text', weight=1)

        self.assertEqual(graph.edges, expected_graph.edges)
        self.assertEqual(graph.edges['This', 'is']['weight'], expected_graph.edges['This', 'is']['weight'])

    def test_create_graph_from_tfidf_matrix(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([self.text])
        graph = create_graph_from_tfidf_matrix(tfidf_matrix)

        expected_graph = nx.DiGraph()
        expected_graph.add_edge('This', 'is', weight=1.0)
        expected_graph.add_edge('is', 'a', weight=1.0)
        expected_graph.add_edge('a', 'test', weight=1.0)
        expected_graph.add_edge('test', 'text', weight=1.0)

        self.assertEqual(graph.edges, expected_graph.edges)
        self.assertEqual(graph.edges['This', 'is']['weight'], expected_graph.edges['This', 'is']['weight'])

    def test_create_character_graph(self):
        graph = create_character_graph(self.text)

        expected_graph = nx.DiGraph()
        expected_graph.add_node('T', frequency=1)
        expected_graph.add_node('h', frequency=1)
        expected_graph.add_node('i', frequency=2)
        expected_graph.add_node('s', frequency=3)
        expected_graph.add_node('a', frequency=1)
        expected_graph.add_node('t', frequency=2)
        expected_graph.add_node('e', frequency=2)
        expected_graph.add_node('x', frequency=1)

        self.assertEqual(graph.nodes, expected_graph.nodes)
        self.assertEqual(graph.nodes['s']['frequency'], expected_graph.nodes['s']['frequency'])

    def test_create_word_graph(self):
        graph = create_word_graph(self.text)

        expected_graph = nx.DiGraph()
        expected_graph.add_node('This', frequency=1)
        expected_graph.add_node('is', frequency=1)
        expected_graph.add_node('a', frequency=1)
        expected_graph.add_node('test', frequency=1)
        expected_graph.add_node('text', frequency=1)

        self.assertEqual(graph.nodes, expected_graph.nodes)
        self.assertEqual(graph.nodes['This']['frequency'], expected_graph.nodes['This']['frequency'])

    def test_create_stopword_punctuation_graph(self):
        graph = create_stopword_punctuation_graph(self.text)

        expected_graph = nx.DiGraph()
        expected_graph.add_node('is', frequency=1, type='stopword')
        expected_graph.add_node('a', frequency=1, type='stopword')
        expected_graph.add_node('test', frequency=1)
        expected_graph.add_node('text', frequency=1)

        self.assertEqual(graph.nodes, expected_graph.nodes)
        self.assertEqual(graph.nodes['is']['type'], expected_graph.nodes['is']['type'])

    def test_convert_graph_to_numpy(self):
        graph = create_character_graph(self.text)
        numpy_graph, node_indices = convert_graph_to_numpy(graph)

        expected_numpy_graph = np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0, 0],
                                         [1, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 1, 0, 0, 0, 1],
                                         [0, 0, 0, 0, 0, 0, 1, 0]])
        expected_node_indices = {'T': 0, 'h': 1, 'i': 2, 's': 3, 'a': 4, 't': 5, 'e': 6, 'x': 7}

        np.testing.assert_array_equal(numpy_graph, expected_numpy_graph)
        self.assertEqual(node_indices, expected_node_indices)

    def test_convert_numpy_to_umap(self):
        graph = create_character_graph(self.text)
        numpy_graph, _ = convert_graph_to_numpy(graph)
        umap_embedding = convert_numpy_to_umap((numpy_graph, {}))

        self.assertEqual(umap_embedding.shape, (8, 2))

    def test_convert_graphs_to_numpy(self):
        graph1 = create_character_graph(self.text)
        graph2 = create_word_graph(self.text)
        graphs = {'character_graph': graph1, 'word_graph': graph2}
        numpy_graphs = convert_graphs_to_numpy(graphs)

        expected_numpy_graph1 = np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 0, 1, 0]])
        expected_numpy_graph2 = np.array([[0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0],
                                          [1, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 1]])
        expected_node_indices1 = {'T': 0, 'h': 1, 'i': 2, 's': 3, 'a': 4, 't': 5, 'e': 6, 'x': 7}
        expected_node_indices2 = {'This': 0, 'is': 1, 'a': 2, 'test': 3, 'text': 4}

        np.testing.assert_array_equal(numpy_graphs['character_graph'][0], expected_numpy_graph1)
        np.testing.assert_array_equal(numpy_graphs['word_graph'][0], expected_numpy_graph2)
        self.assertEqual(numpy_graphs['character_graph'][1], expected_node_indices1)
        self.assertEqual(numpy_graphs['word_graph'][1], expected_node_indices2)


if __name__ == '__main__':
    unittest.main()

import pytest
import numpy as np

from psyneulink.compositions.composition import Graph, Vertex, Composition
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection


class TestGraph:

    class DummyComponent:

        def __init__(self):
            pass

    def test_copy(self):

        g1 = Graph()
        vertices = [Vertex(TestGraph.DummyComponent()) for i in range(5)]

        for i in range(len(vertices)):
            g1.add_vertex(vertices[i])
            # each vertex has previous vertex as parent and next vertex as child
            g1.connect_vertices(vertices[(i - 1) % len(vertices)], vertices[i])

        g2 = g1.copy()

        assert len(g1.vertices) == len(g2.vertices)
        assert len(g1.comp_to_vertex) == len(g2.comp_to_vertex)

        for i in range(len(g2.vertices)):
            assert g2.vertices[i].parents == [g2.vertices[(i - 1) % len(g2.vertices)]]
            assert g2.vertices[i].children == [g2.vertices[(i + 1) % len(g2.vertices)]]

            assert g1.vertices[i] != g2.vertices[i]
            assert g1.vertices[i].component == g2.vertices[i].component

class TestCycles:

    def test_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        C2 = ProcessingMechanism(name="C2")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D, E])
        comp.add_linear_processing_pathway([D, C2, B])

        comp.run(inputs={A: [1.0]})

        print(comp.scheduler_processing.consideration_queue)

    def test_simple_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        comp.add_linear_processing_pathway([D, MappingProjection(matrix=4.0), B])

        cycle_nodes = [B, C, D]
        for cycle_node in cycle_nodes:
            cycle_node.output_states[0].value = [1.0]

        comp.run(inputs={A: [1.0]})
        expected_values = {A: 1.0,
                           B: 5.0,
                           C: 2.0,
                           D: 3.0,
                           E: 3.0}

        for node in expected_values:
            assert np.allclose(expected_values[node], node.value)

        comp.run(inputs={A: [1.0]})
        expected_values_2 = {A: 1.0,
                             B: 13.0,
                             C: 10.0,
                             D: 6.0,
                             E: 6.0}

        for node in expected_values:
            assert np.allclose(expected_values_2[node], node.value)

    def test_two_loops(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        C2 = ProcessingMechanism(name="C2")
        C3 = ProcessingMechanism(name="C3")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D, E])
        comp.add_linear_processing_pathway([D, C2, B])
        comp.add_linear_processing_pathway([D, C3, B])

        comp.run(inputs={A: [1.0]})

        print(comp.scheduler_processing.consideration_queue)



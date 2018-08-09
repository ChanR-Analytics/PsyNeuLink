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

    def test_loop_with_extra_node(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        C2 = ProcessingMechanism(name="C2")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()

        cycle_nodes = [B, C, D, C2]
        for cycle_node in cycle_nodes:
            cycle_node.output_states[0].value = [1.0]

        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, D, MappingProjection(matrix=5.0), E])
        comp.add_linear_processing_pathway([D, MappingProjection(matrix=3.0), C2, MappingProjection(matrix=4.0), B])

        comp.run(inputs={A: [1.0]})
        expected_values = {A: 1.0,
                           B: 5.0,
                           C: 2.0,
                           D: 2.0,
                           C2: 3.0,
                           E: 2.0}

        for node in expected_values:
            assert np.allclose(expected_values[node], node.value)

        comp.run(inputs={A: [1.0]})
        expected_values_2 = {A: 1.0,
                             B: 13.0,
                             C: 10.0,
                             D: 10.0,
                             C2: 6.0,
                             E: 10.0}

        for node in expected_values:
            assert np.allclose(expected_values_2[node], node.value)

    def test_two_overlapping_loops(self):
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

        assert comp.scheduler_processing.consideration_queue[0] == {A}
        assert comp.scheduler_processing.consideration_queue[1] == {B, C, D, C2, C3}
        assert comp.scheduler_processing.consideration_queue[2] == {E}

    def test_two_separate_loops(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        L1 = ProcessingMechanism(name="L1")
        L2 = ProcessingMechanism(name="L2")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")
        F = ProcessingMechanism(name="F")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D, E, F])
        comp.add_linear_processing_pathway([E, L1, D])
        comp.add_linear_processing_pathway([C, L2, B])

        comp.run(inputs={A: [1.0]})

        assert comp.scheduler_processing.consideration_queue[0] == {A}
        assert comp.scheduler_processing.consideration_queue[1] == {C, L2, B}
        assert comp.scheduler_processing.consideration_queue[2] == {E, L1, D}
        assert comp.scheduler_processing.consideration_queue[3] == {F}

    def test_two_paths_converge(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D])
        comp.add_linear_processing_pathway([E, D])

        comp.run(inputs={A: 1.0,
                         E: 1.0})

        assert comp.scheduler_processing.consideration_queue[0] == {A, E}
        assert comp.scheduler_processing.consideration_queue[1] == {B}
        assert comp.scheduler_processing.consideration_queue[2] == {C}
        assert comp.scheduler_processing.consideration_queue[3] == {D}

    def test_diverge_and_reconverge(self):
        S = ProcessingMechanism(name="START")
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([S, A, B, C, D])
        comp.add_linear_processing_pathway([S, E, D])

        comp.run(inputs={S: 1.0})

        assert comp.scheduler_processing.consideration_queue[0] == {S}
        assert comp.scheduler_processing.consideration_queue[1] == {A, E}
        assert comp.scheduler_processing.consideration_queue[2] == {B}
        assert comp.scheduler_processing.consideration_queue[3] == {C}
        assert comp.scheduler_processing.consideration_queue[4] == {D}

    def test_diverge_and_reconverge_2(self):
        S = ProcessingMechanism(name="START")
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")
        F = ProcessingMechanism(name="F")
        G = ProcessingMechanism(name="G")

        comp = Composition()
        comp.add_linear_processing_pathway([S, A, B, C, D])
        comp.add_linear_processing_pathway([S, E, F, G, D])

        comp.run(inputs={S: 1.0})

        assert comp.scheduler_processing.consideration_queue[0] == {S}
        assert comp.scheduler_processing.consideration_queue[1] == {A, E}
        assert comp.scheduler_processing.consideration_queue[2] == {B, F}
        assert comp.scheduler_processing.consideration_queue[3] == {C, G}
        assert comp.scheduler_processing.consideration_queue[4] == {D}

    def test_figure_eight(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C1 = ProcessingMechanism(name="C1")
        D1 = ProcessingMechanism(name="D1")
        C2 = ProcessingMechanism(name="C2")
        D2 = ProcessingMechanism(name="D2")

        comp = Composition()

        comp.add_linear_processing_pathway([A, B])
        comp.add_linear_processing_pathway([B, C1, D1])
        comp.add_linear_processing_pathway([B, C2, D2])
        comp.add_linear_processing_pathway([D1, B])
        comp.add_linear_processing_pathway([D2, B])

        assert comp.scheduler_processing.consideration_queue[0] == {A}
        assert comp.scheduler_processing.consideration_queue[1] == {B, C1, D1, C2, D2}

    def test_many_loops(self):

        comp = Composition()

        start = ProcessingMechanism(name="start")
        expected_consideration_sets = [{start}]
        for i in range(10):
            A = ProcessingMechanism(name='A'+str(i))
            B = ProcessingMechanism(name='B'+str(i))
            C = ProcessingMechanism(name='C'+str(i))
            D = ProcessingMechanism(name='D'+str(i))

            comp.add_linear_processing_pathway([start, A, B, C, D])
            comp.add_linear_processing_pathway([C, B])

            expected_consideration_sets.append({A})
            expected_consideration_sets.append({B, C})
            expected_consideration_sets.append({D})

            start = D

        for i in range(len(comp.scheduler_processing.consideration_queue)):
            assert comp.scheduler_processing.consideration_queue[i] == expected_consideration_sets[i]
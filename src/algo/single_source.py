from abc import ABC, abstractmethod
from typing import Iterable
from pygraphblas import Matrix
from pygraphblas.types import BOOL

from src.grammar.cnf_grammar import CnfGrammar
from src.graph.label_graph import LabelGraph


class SingleSourceIndex:
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        self.graph = graph
        self.grammar = grammar
        self.sources = LabelGraph()
        self.nonterms = LabelGraph()

    def init_simple_rules(self):
        for l, r in self.grammar.simple_rules:
            self.nonterms[l] = self.graph[r].dup()


class SingleSourceSolver(ABC):
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        self.graph = graph
        self.grammar = grammar

    @abstractmethod
    def solve(self, sources_vertices: Iterable) -> Matrix:
        pass


class SingleSourceAlgoSmart(SingleSourceSolver):
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        super().__init__(graph, grammar)
        self.index = SingleSourceIndex(graph, grammar)

    def solve(self, sources_vertices: Iterable) -> Matrix:
        # Use self.index
        pass


class SingleSourceAlgoBrute(SingleSourceSolver):
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        super().__init__(graph, grammar)

    def solve(self, sources_vertices: Iterable) -> Matrix:
        index = SingleSourceIndex(self.graph, self.grammar)

        index.init_simple_rules()

        for v in sources_vertices:
            index.sources[index.grammar.start_nonterm][v, v] = True

        changed = True

        while changed:
            for l, r1, r2 in index.grammar.complex_rules:
                old_nnz = index.nonterms[l].nvals

                i_l, j_l, vs_l = index.sources[l].to_lists()
                for v, to in list(zip(i_l, j_l)):
                    index.sources[r1][to, to] = True

                tmp = Matrix.sparse(BOOL, index.graph.matrices_size,
                                    index.graph.matrices_size)
                tmp = index.sources[l] @ index.nonterms[r1]

                i_r2, j_r2, vs_r2 = tmp.to_lists()
                for v, to in list(zip(i_r2, j_r2)):
                    index.sources[r2][to, to] = True

                index.nonterms[l] += tmp @ index.nonterms[r2]

                new_nnz = index.nonterms[l].nvals
                changed = not old_nnz == new_nnz

        return index.nonterms[index.grammar.start_nonterm]

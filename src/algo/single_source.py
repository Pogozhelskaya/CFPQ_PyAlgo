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
            self.nonterms[l] += self.graph[r]


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


class SingleSourceAlgoOpt(SingleSourceSolver):
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        super().__init__(graph, grammar)
        self.index = SingleSourceIndex(graph, grammar)

    def solve(self, sources_vertices: Iterable) -> Matrix:
        cur_index = SingleSourceIndex(self.graph, self.grammar)
        # Initialize simple rules
        cur_index.init_simple_rules()
        # Initialize source matrices masks
        for v in sources_vertices:
            cur_index.sources[self.index.grammar.start_nonterm][v, v] = True
        SingleSourceAlgoOpt.__update_sources(
            self.index.sources[self.index.grammar.start_nonterm],
            cur_index.sources[self.index.grammar.start_nonterm],
            self.index.sources[self.index.grammar.start_nonterm])
        # Create temporary matrix
        tmp = Matrix.sparse(BOOL,
                            cur_index.graph.matrices_size,
                            cur_index.graph.matrices_size)
        # Algo's body
        changed = True
        while changed:
            changed = False
            # Iterate through all complex rules
            for l, r1, r2 in self.index.grammar.complex_rules:
                # Number of instances before operation
                old_nnz = cur_index.nonterms[l].nvals

                # l -> r1 r2 ==> l += (l_src * r1) * r2 =>

                # 1) r1_src += {(j, j) : (i, j) \in l_src}
                SingleSourceAlgoOpt.__update_sources(cur_index.sources[l],
                                                     cur_index.sources[r1],
                                                     self.index.sources[r1])

                # 2) tmp = l_src * r1
                tmp = cur_index.sources[l] @ cur_index.nonterms[r1]

                # 3) r2_src += {(j, j) : (i, j) \in tmp}
                SingleSourceAlgoOpt.__update_sources(tmp,
                                                     cur_index.sources[r2],
                                                     self.index.sources[r2])

                # 4) l += tmp * r2
                cur_index.nonterms[l] += tmp @ cur_index.nonterms[r2]

                # Number of instances after operation
                new_nnz = cur_index.nonterms[l].nvals

                # Update changed flag
                changed |= not old_nnz == new_nnz

        for nonterm in self.grammar.nonterms:
            self.index.nonterms[nonterm] += cur_index.nonterms[nonterm]
            self.index.sources[nonterm] += cur_index.sources[nonterm]
        return self.index.nonterms[self.index.grammar.start_nonterm]

    @staticmethod
    def __update_sources(src: Matrix, dst: Matrix, msk: Matrix):
        i_src, j_src, v_src = src.to_lists()
        for k in range(len(j_src)):
            if v_src[k] is True:
                dst[j_src[k], j_src[k]] = True

        for v in msk.ncols:
            if dst[v, v] is True & msk[v, v] is True:
                dst[v, v] = False

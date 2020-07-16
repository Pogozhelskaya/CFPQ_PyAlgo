from pygraphblas import binaryop, Matrix

from src.grammar.cnf_grammar import CnfGrammar
from src.graph.label_graph import LabelGraph

from pygraphblas.demo.gviz import draw, draw_op
from pygraphblas import *

def matrix_base_algo(g: LabelGraph, grammar: CnfGrammar):
    m = LabelGraph()
    for l, r in grammar.simple_rules:
        m[l] += g[r]
    changed = True
    while changed:
        for l, r1, r2 in grammar.complex_rules:
            old_nnz = m[l].nvals
            m[l] += m[r1] @ m[r2]
           # dg = draw_op(m[r1], '@', m[r2],m[r1] @ m[r2] )
           # dg.view()
            new_nnz = m[l].nvals
            changed = not old_nnz == new_nnz
    return m

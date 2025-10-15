from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Expr:
    def __add__(self, other: Expr):
        return Add(self, other)

    def __mul__(self, other: Expr):
        return Mul(self, other)

@dataclass(frozen=True)
class Vector(Expr):
    name: str

    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class Add(Expr):
    a: Expr
    b: Expr

    def __repr__(self):
        return f"({self.a} + {self.b})"

@dataclass(frozen=True)
class Mul(Expr):
    a: Expr
    b: Expr

    def __repr__(self):
        return f"({self.a} * {self.b})"

class Node:
    def __init__(self, expr: Expr | None):
        # None -> Empty
        assert expr is None or isinstance(expr, Expr)
        self.expr = expr
        self.children = set()

    def add_child(self, node: "Node"):
        assert isinstance(node, Node)
        self.children.add(node)

    def __repr__(self):
        if self.expr is None:
            return "Empty"
        else:
            expr_repr = repr(self.expr)
        if not self.children:
            return f"Node({expr_repr})"
        return f"Node({expr_repr}, children={self.children})"

    # eq and hash required for set() to operate correctly
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.expr == other.expr and self.children == other.children

    def __hash__(self):
        # convert children set to frozenset for hashing
        return hash((self.expr, frozenset(self.children)))

def get_edges(expr: Expr | None):
    edges = []

    if expr is None:
        return edges

    def recurse(e: Expr):
        match e:
            case Vector(name):
                edges.append(e)
            case Add(a, b) | Mul(a, b):
                recurse(a)
                recurse(b)
            case _:
                raise TypeError(f"Unexpected expression type: {type(e)}")

    recurse(expr)
    return edges

def remove_and_simplify(e : Expr, expr : Expr):
    if e == expr:
        return None
    match expr:
        case Vector(name):
            return expr
        case Add(a, b):
            a = remove_and_simplify(e, a)
            b = remove_and_simplify(e, b)
            if a is None:
                return b
            elif b is None:
                return a
            else:
                return a + b
        case Mul(a, b):
            a = remove_and_simplify(e, a)
            b = remove_and_simplify(e, b)
            if a is None or b is None:
                return None
            else:
                return a * b
        case _:
            raise TypeError(f"Unexpected expression type: {type(e)}")

# Burrito lattice construction algorithm (Figure 12, page 12)
# TODO: this builds a tree because we don't deduplicate.
# That should be fine, just slow for large expressions
def build_lattice(expr : Expr):
    point = Node(expr)
    edges = get_edges(expr)

    for e in edges:
        s = remove_and_simplify(e, expr)
        l = build_lattice(s)
        point.add_child(l)
    return point

def get_sorted_args(expr : Expr):
    edges = get_edges(expr)
    # Extract unique Vector objects and sort by name
    vectors = sorted({v.name for v in edges})
    return vectors 

# Vector expression to compile, and flag to enable compute or counting
def compile(expr : Expr, func_name : str, compute : bool = True):
    vectors = get_sorted_args(expr)

    indent1 = "  "
    indent2 = "    "
    indent3 = "      "

    print("template<typename index_t, typename value_t>")
    print(f"void {func_name}(")

    # For each argument in args, print a const SparseVector& with an underscore added
    for vec in vectors:
        print(f"{indent1}const SparseVector<index_t, value_t> &_{vec},")

    # Then add an output and close the parens
    if compute:
        print(f"{indent1}SparseVector<index_t, value_t> &output)\n{{")
    else:
        # array of counts. TODO: global atomic?
        print(f"{indent1}index_t *nnzs) {{")
        # local count
        print(f"{indent1}index_t nnz_count = 0;")

    # TODO: this is where the load balancing strategy should go!
    # First, declare start and end indices
    for vec in vectors:
        print(f"{indent1}index_t idx_{vec} = 0, end_{vec} = _{vec}.nnz;")

    # TODO: for parallel, need some other way of computing output iterator.
    if compute:
        print(f"{indent1}index_t idx_output = 0;")

    # Now build the loop(s)
    lattice = build_lattice(expr)

    def compile_loop(point : Node):
        if point.expr is None:
            return
        iters = get_sorted_args(point.expr)
        terms = [f"(idx_{s} < end_{s})" for s in iters]
        while_cond = " && ".join(terms)

        print(f"{indent1}while({while_cond}) {{")

        for vec in iters:
            print(f"{indent2}index_t crd_{vec} = _{vec}.indices[idx_{vec}];")

        # build a min on all crd_{vec} using binary min
        min_expr = f"crd_{iters[0]}"
        for vec in iters[1:]:
            min_expr = f"std::min({min_expr}, crd_{vec})"
        print(f"{indent2}index_t crd = {min_expr};")

        # Now perform casework. Consider this node and all children.
        compile_if(point, True)

        for child in point.children:
            if child.expr is not None:
                compile_if(child, False)

        # Now step forward iterators
        for vec in iters:
            print(f"{indent2}idx_{vec} += (index_t)(crd_{vec} == crd);")

        print(f"{indent1}}}")

        for child in point.children:
            if child.expr is not None:
                compile_loop(child)

    def compile_if(point : Node, first : bool):
        iters = get_sorted_args(point.expr)
        terms = [f"(crd_{s} == crd)" for s in iters]
        if_cond = " && ".join(terms)

        if not first:
            print(f"{indent2}else if ({if_cond}) {{")
        else:
            print(f"{indent2}if ({if_cond}) {{")

        if not compute:
            print(f"{indent3}nnz_count++;")
        else:
            print(f"{indent3}output.indices[idx_output] = crd;")
            for vec in iters:
                print(f"{indent3}value_t {vec} = _{vec}.values[idx_{vec}];")
            print(f"{indent3}output.values[idx_output++] = {point.expr};")

        print(f"{indent2}}}")


    compile_loop(lattice)

    if not compute:
        # TODO: this should be thread id
        print(f"{indent1}nnzs[0] = nnz_count;")

    print("}")


if __name__ == "__main__":
    a = Vector("a")
    b = Vector("b")
    c = Vector("c")

    expr = a * b
    compile(expr, "mul_nnz_count", False)
    compile(expr, "mul_compute", True)

    expr = (a + b) * c
    compile(expr, "plus_mul_nnz_count", False)
    compile(expr, "plus_mul_compute", True)


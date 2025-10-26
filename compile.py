from __future__ import annotations
from dataclasses import dataclass
from typing import List

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
# lb is a list of the vectors to load balance with respect to
# lb must be in the order that we expect to read from the partitions array
def compile(expr : Expr, lb : List[str], func_name : str, compute : bool = True):
    vectors = get_sorted_args(expr)

    indent1 = "  "
    indent2 = "    "
    indent3 = "      "

    print(f"/** {expr} */")
    print("template<typename index_t, typename value_t>")
    if not compute:
        func_name += "_precompute"
    print(f"void {func_name}(")

    print(f"{indent1}const index_t *partitions,")
    if compute:
        print(f"{indent1}const index_t *output_offset,")

    # For each argument in args, print a const SparseVector& with an underscore added
    for vec in vectors:
        print(f"{indent1}const SparseVector<index_t, value_t> &_{vec},")

    # Then add an output and close the parens
    if compute:
        print(f"{indent1}SparseVector<index_t, value_t> &output)\n{{")
    else:
        # array of counts. TODO: global atomic?
        print(f"{indent1}index_t *nnzs)\n{{")
        # local count
        print(f"{indent1}index_t nnz_count = 0;")

    print(f"{indent1}const int n_threads = gridDim.x * blockDim.x;")
    print(f"{indent1}const int tid = blockIdx.x * blockDim.x + threadIdx.x;")
    print()

    # First, declare start and end indices
    for vec in vectors:
        print(f"{indent1}index_t idx_{vec} = 0, end_{vec} = _{vec}.nnz - 1;")

    # Now get bounds via load balancing.

    def lower_bound(name, offset):
        assert name in lb
        if offset is None:
            # Just read from register
            return f"idx_{name}"
        else:
            # Need to read from `partitons`
            i = lb.index(name)
            return f"partitions[{i} * n_threads + (tid + {offset})]"

    def upper_bound(name, offset):
        assert name in lb
        if offset is None:
            # Just read from register
            return f"end_{name}"
        else:
            # Need to read from `partitons`
            i = lb.index(name)
            return f"partitions[{i} * n_threads + (tid + 1 + {offset})]"

    # First, grab all bounds from
    for (i, name) in enumerate(lb):
        print(f"{indent1}idx_{name} = partitions[{i} * n_threads + tid];")
        print(f"{indent1}end_{name} = partitions[{i} * n_threads + tid + 1];")

    bounded = set(lb)

    def min_bound(lb0, lb1):
        if lb0 is None:
            return lb1
        elif lb1 is None:
            return lb0
        else:
            return f"std::min({lb0}, {lb1})"
    def max_bound(lb0, lb1):
        if lb0 is None:
            return lb1
        elif lb1 is None:
            return lb0
        else:
            return f"std::max({lb0}, {lb1})"

    def get_coordinate_bounds(expr, wide):
        match expr:
            case Vector(name):
                if name in bounded:
                    lb = f"_{name}.indices[idx_{name}]"
                    # For now, wide just takes the "rest" of the RHS chunk.
                    wide_ub = f"_{name}.indices[std::min(end_{name} + 1, _{name}.nnz - 1)]"
                    narrow_ub = f"_{name}.indices[end_{name}]"
                    ub = wide_ub if wide else narrow_ub
                    return lb, ub
                else:
                    return None, None
            case Add(a, b):
                a_lb, a_ub = get_coordinate_bounds(a, wide)
                b_lb, b_ub = get_coordinate_bounds(b, wide)
                if a_lb is None or a_ub is None or b_lb is None or b_ub is None:
                    return None, None
                # The bounds of an add are the union of bounds
                return min_bound(a_lb, b_lb), max_bound(a_ub, b_ub)
            case Mul(a, b):
                a_lb, a_ub = get_coordinate_bounds(a, wide)
                b_lb, b_ub = get_coordinate_bounds(b, wide)
                # if (a_lb is None and b_lb is None) or (a_ub is None and b_ub is None):
                if a_lb is None or a_ub is None or b_lb is None or b_ub is None:
                    return None, None
                # The bounds of a mul are the intersection of bounds
                # BUT, if wide is true, we need the union of bounds!
                if not wide:
                    # intersection
                    return max_bound(a_lb, b_lb), min_bound(a_ub, b_ub)
                else:
                    # union
                    return min_bound(a_lb, b_lb), max_bound(a_ub, b_ub)
            case _:
                raise TypeError(f"Unexpected expression type: {type(e)}")

    # Now, for any vector that is not load balanced, we need to *propagate* bounds.
    def propagate_bounds(expr, lb, ub):
        match expr:
            case Vector(name):
                if name not in bounded:
                    assert lb is not None, f"Lower bound is None for vector '{name}'"
                    assert ub is not None, f"Upper bound is None for vector '{name}'"
                    print(f"{indent1}index_t crd_lb_{name} = {lb};")
                    print(f"{indent1}index_t crd_ub_{name} = {ub};")
                    print(f"{indent1}idx_{name} = (tid == 0) ? 0 : upper_bound({name}, crd_lb_{name});")
                    print(f"{indent1}end_{name} = lower_bound({name}, crd_ub_{name});")
                    bounded.add(name)
            case Add(a, b):
                a_lb, a_ub = get_coordinate_bounds(a, wide=True)
                b_lb, b_ub = get_coordinate_bounds(b, wide=True)
                # [lb, ub] is all we need, so clamp the union of
                # a and b to those.
                new_lb = max_bound(min_bound(a_lb, b_lb), lb)
                new_ub = min_bound(max_bound(a_ub, b_ub), ub)
                if a_lb is None or a_ub is None:
                    # Propagate through a
                    propagate_bounds(a, new_lb, new_ub)
                if b_lb is None or b_ub is None:
                    # Propagate through a
                    propagate_bounds(b, new_lb, new_ub)
            case Mul(a, b):
                a_lb, a_ub = get_coordinate_bounds(a, wide=False)
                b_lb, b_ub = get_coordinate_bounds(b, wide=False)
                # [lb, ub] is all we need, so clamp the intersection
                # of a and b to those.
                new_lb = max_bound(max_bound(a_lb, b_lb), lb)
                new_ub = min_bound(min_bound(a_ub, b_ub), ub)
                if a_lb is None or a_ub is None:
                    # Propagate through a
                    propagate_bounds(a, new_lb, new_ub)
                if b_lb is None or b_ub is None:
                    # Propagate through a
                    propagate_bounds(b, new_lb, new_ub)
            case _:
                raise TypeError(f"Unexpected expression type: {type(e)}")

    propagate_bounds(expr, None, None)

    if compute:
        print(f"{indent1}index_t idx_output = output_offset[tid];")

    # Now build the loop(s)
    lattice = build_lattice(expr)

    def compile_loop(point : Node):
        if point.expr is None:
            return
        iters = get_sorted_args(point.expr)
        terms = [f"(idx_{s} <= end_{s})" for s in iters]
        while_cond = " && ".join(terms)

        print(f"\n{indent1}while({while_cond}) {{")

        for vec in iters:
            print(f"{indent2}index_t crd_{vec} = _{vec}.indices[idx_{vec}];")

        # build a min on all crd_{vec} using binary min
        min_expr = f"crd_{iters[0]}"
        for vec in iters[1:]:
            min_expr = f"std::min({min_expr}, crd_{vec})"
        print(f"{indent2}index_t crd = {min_expr};")
        print()

        # Now perform casework. Consider this node and all children.
        compile_if(point, True)

        for child in point.children:
            if child.expr is not None:
                compile_if(child, False)
        
        print(f"{indent2}}}")

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
            print(f"{indent2}}} else if ({if_cond}) {{")
        else:
            print(f"{indent2}if ({if_cond}) {{")

        if not compute:
            print(f"{indent3}nnz_count++;")
        else:
            print(f"{indent3}output.indices[idx_output] = crd;")
            for vec in iters:
                print(f"{indent3}value_t {vec} = _{vec}.values[idx_{vec}];")
            print(f"{indent3}output.values[idx_output++] = {point.expr};")


    compile_loop(lattice)

    if not compute:
        # TODO: this should be thread id
        print(f"{indent1}nnzs[tid] = nnz_count;")

    print("}")


if __name__ == "__main__":
    a = Vector("a")
    b = Vector("b")
    c = Vector("c")

    # expr = a * b
    # compile(expr, ["a"], "mul_lb_a", False)
    # print()
    # compile(expr, ["a"], "mul_lb_a", True)
    # print()

    # compile(expr, ["a", "b"], "mul_lb_ab", False)
    # print()
    # compile(expr, ["a", "b"], "mul_lb_ab", True)
    # print()

    # expr = a + b
    # compile(expr, ["a"], "add_lb_a", False)
    # print()
    # compile(expr, ["a"], "add_lb_a", True)
    # print()

    expr = a + b * c
    compile(expr, ["b", "c"], "a_bc", False)
    print()
    compile(expr, ["b", "c"], "a_bc", True)
    print()


    expr = (a + b) * c
    compile(expr, ["b", "c"], "ab_c", False)
    print()
    compile(expr, ["b", "c"], "ab_c", True)
    print()

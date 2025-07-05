from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals_plus_epsilon = list(vals)
    vals_plus_epsilon[arg] = vals_plus_epsilon[arg] + epsilon

    vals_minus_epsilon = list(vals)
    vals_minus_epsilon[arg] = vals_minus_epsilon[arg] - epsilon

    return (f(*vals_plus_epsilon) - f(*vals_minus_epsilon)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.

    sorted_nodes = deque()

    permanent_mark = set()
    temporary_mark = set()

    def visit_node(node: Variable):
        if node.unique_id in permanent_mark or node.is_constant():
            return
        if node.unique_id in temporary_mark:
            raise ValueError("Graph has at least one cycle")

        temporary_mark.add(node.unique_id)

        for parent in node.parents:
            visit_node(parent)

        permanent_mark.add(node.unique_id)
        sorted_nodes.appendleft(node)

    visit_node(variable)

    return list(sorted_nodes)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.

    ordered_que = topological_sort(variable)

    current_derivatives: dict[str, tuple[Variable, Any]] = {
        variable.unique_id: (variable, deriv)
    }
    
    for node in ordered_que:
        _, d_output = current_derivatives.get(node.unique_id)

        if d_output is None:
            continue

        if node.is_leaf():
            node.accumulate_derivative(d_output)
            continue

        partial_derivatives = node.chain_rule(d_output)

        for variable, derivative in partial_derivatives:
            if variable.unique_id not in current_derivatives:
                current_derivatives[variable.unique_id] = (variable, derivative)
            else:
            # Accounting for multiple uses of a node as input:
            # If a variable is used in multiple inputs (e.g., f(g(x), g(x))),
            # then by the multivariate chain rule, we must accumulate its
            # partial derivatives from all paths.
                current_derivatives[variable.unique_id] = (
                    current_derivatives[variable.unique_id][0],
                    current_derivatives[variable.unique_id][1] + derivative,
                )


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values


if __name__ == "__main__":
    from operators import id

    d = central_difference(id, 5, arg=0)

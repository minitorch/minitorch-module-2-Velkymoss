from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train_deprecated(self) -> None:
        """Set the mode of this module and all descendent modules to `train`."""
        # TODO: Implement for Task 0.4.
        self.training = True
        nodes_to_process = self._modules.values()
        future_nodes = []
        while nodes_to_process:
            for node in nodes_to_process:
                node.training = True
                future_nodes.extend(node._modules.values())
            nodes_to_process = future_nodes
            future_nodes = []

    def train(self) -> None:
        """Set the mode of this module and all descendent modules to `train`.
        
        Traverses tree level by level (BFS)
        """
        # TODO: Implement for Task 0.4.
        self.training = True
        que = deque(self._modules.values())
        while que:
            node = que.popleft()
            node.training = True
            que.extend(node._modules.values())

    def eval(self) -> None:
        """Set the mode of this module and all descendent modules to `eval`.

        Traverses tree level by level (BFS)
        """
        # TODO: Implement for Task 0.4.
        self.training = False
        que = deque(self._modules.values())
        while que:
            node = que.popleft()
            node.training = False
            que.extend(node._modules.values())

    def named_parameters_deprecated(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.

        """
        # TODO: Implement for Task 0.4.
        parameters = list(self._parameters.items())

        que = deque(self._modules.items())

        # list of lists with module_name, number_of_children + 1 elements
        path = []

        while que:
            module_name, node = que.popleft()

            # create the module path
            joined_path = ""
            if path:
                path.append([module_name, len(node._modules) + 1])
                for subpath in path:
                    joined_path = joined_path + subpath[0] + "."
            else:
                path.append([module_name, len(node._modules) + 1])
                joined_path = path[0][0] + "."

            # at parameter name to module paths
            named_parameters = [
                (joined_path + name, parameter)
                for name, parameter in node._parameters.items()
            ]

            # update path
            if path:
                new_path = []
                for i in range(len(path)):
                    path[i][1] -= 1

                    if path[i][1] > 0:
                        new_path.append(path[i])
                path = new_path

            parameters.extend(named_parameters)

            # add children to que so they are processed next
            que.extendleft(node._modules.items())

        return parameters

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Traverses tree level by level (BFS)

        Performs cycle detection

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.

        """
        # add parameters of root module
        parameters = list(self._parameters.items())
        # que: [(path, Module), ...]
        que = deque([(name, module) for name, module in self._modules.items()])
        # keep track of ids of processed Modules
        visited = set()

        while que:
            prefix, module = que.popleft()

            # check module id
            if id(module) in visited:
                continue
            visited.add(id(module))

            # add parameters of current modules to parameters
            for param_name, param in module._parameters.items():
                full_name = prefix + "." + param_name
                parameters.append((full_name, param))

            # create path for children and append to que
            for child_key, child_module in module._modules.items():
                child_prefix = prefix + "." + child_key
                que.append((child_prefix, child_module))

        return parameters

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        # TODO: Implement for Task 0.4.
        parameters = list(self._parameters.values())
        que = deque(self._modules.values())
        while que:
            node = que.popleft()
            que.extend(node._modules.values())
            parameters.extend(node._parameters.values())
        return parameters

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)

from __future__ import annotations

import ast
import re

import numpy as np
from sympy import Function, Symbol, lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: Custom SymPy function for matrix multiplication
matmul = Function("matmul")

#: The alias substituted for the ``in`` keyword, which is not a legal
#: Python identifier. Permitted by name because the gate otherwise
#: rejects dunders, and this one is ours.
_IN_ALIAS = "__in_alias__"

#: AST nodes an arithmetic expression may contain. An allow-list rather
#: than a deny-list: a deny-list has to be revisited every time Python
#: grows syntax, and the one that is forgotten is the one that gets used.
_ALLOWED_NODES = (
    ast.Expression,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.MatMult,
    ast.UAdd,
    ast.USub,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _reject_unsafe(expression: str, original: str) -> None:
    """Refuse an expression that is not plain arithmetic.

    ``parse_expr`` is not a sandbox, and SymPy does not claim it is:
    :func:`sympy.parsing.sympy_parser.eval_expr` is literally
    ``eval(code, global_dict, local_dict)``. Two things follow, both
    measured rather than assumed:

    * **Attribute access on a literal evaluates.** ``().__class__``
      returns the real ``tuple`` type, which is the first rung of the
      ``__subclasses__()`` ladder to arbitrary code. Emptying the
      namespace does not help, because no name is looked up.
    * **Some builtins resolve.** ``__import__('os')`` returns the ``os``
      module, one attribute short of ``os.system``.

    Neither needs a code bundle or an unusual module name: an
    ``Equation`` is an ordinary node and its expression is an ordinary
    string parameter. So a document from an untrusted author is code
    unless something refuses it, and refusing it syntactically is what
    works -- restricting the namespace does not.

    The maths is untouched: what a user writes in an expression is
    names, numbers, operators and function calls, all of which are
    allowed.

    Args:
        expression: The expression to check, after the ``in`` and ``@``
            substitutions, so it parses as Python.
        original: What the author wrote, for the error message.

    Raises:
        ValueError: If the expression contains anything but arithmetic.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(
            f"expression {original!r} is not valid syntax: {e.msg}"
        ) from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"expression {original!r} contains "
                f"{type(node).__name__}, which is not arithmetic. An "
                f"expression may use names, numbers, operators and "
                f"function calls -- attribute access, indexing, "
                f"lambdas and comprehensions are refused because they "
                f"reach Python itself rather than the maths."
            )

        # A string has no meaning in a numeric expression, and it is what
        # every reachable escape needs as its payload.
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float, complex)
        ):
            raise ValueError(
                f"expression {original!r} contains a "
                f"{type(node.value).__name__} literal. Only numbers are "
                f"allowed."
            )

        # Dunders are how the interpreter is addressed: __import__,
        # __builtins__, __class__. None of them is arithmetic.
        if isinstance(node, ast.Name):
            name = node.id
            if name != _IN_ALIAS and name.startswith("__"):
                raise ValueError(
                    f"expression {original!r} refers to {name!r}. Names "
                    f"beginning with '__' address the interpreter, not "
                    f"the maths, and are refused."
                )

        # sin(a) is a call to a name. ().__class__() would be a call to
        # something else, and there is no arithmetic reason for one.
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError(
                f"expression {original!r} calls something other than a "
                f"named function, which is refused."
            )


class Equation(IONode):
    """Mathematical expression evaluation node for data transformation.

    Applies custom mathematical expressions to input data using SymPy.
    Automatically creates input ports from expression variables and compiles
    to optimized NumPy functions. Handles 'in' keyword via internal aliasing.
    """

    #: The input ports are the free variables of the expression, so
    #: there is no port list to declare: ``a+b`` has two inputs and
    #: ``eeg1*2 - eeg2`` has two differently named ones. Naming the
    #: parameter that decides them is the whole truth. Publishing one
    #: expression's answer as the class's port set -- which is what
    #: building an instance to find out did -- is correct for exactly
    #: one expression out of infinitely many.
    INPUT_PORTS_FROM = "expression"

    class Configuration(IONode.Configuration):
        """Configuration class for Equation parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration key constants for the Equation."""

            #: Configuration key for mathematical expression string
            EXPRESSION = "expression"

    def __init__(self, expression: str, **kwargs):
        """Initialize Equation node with mathematical expression.

        Parses expression using SymPy, extracts variables to create input
        ports, and compiles to optimized NumPy function.

        Args:
            expression: Mathematical expression string. Must be valid SymPy
                expression. Variables become input port names. 'in' keyword
                handled via internal aliasing.
            **kwargs: Additional configuration parameters for IONode.

        Raises:
            ValueError: If expression is None or empty.
            SymPy parsing errors: If expression cannot be parsed.
        """
        # Validate that expression is provided
        if expression is None:
            raise ValueError("Expression must be specified.")

        # Handle Python keyword 'in' by replacing with internal alias
        # This allows users to use 'in' as a variable name in expressions
        replaced_expr = re.sub(r"\bin\b", "__in_alias__", expression)

        # Handle matrix multiplication operator '@' by replacing with matmul()
        # This allows users to use Python's @ operator for matrix operations
        replaced_expr = re.sub(
            r"(\w+)\s*@\s*(\w+)", r"matmul(\1, \2)", replaced_expr
        )

        # Create symbol mapping for the 'in' keyword alias and matmul function
        local_dict = {
            "__in_alias__": Symbol("in"),
            "matmul": matmul,
        }

        # Refuse anything that is not arithmetic BEFORE handing it to
        # SymPy, because parse_expr evaluates what it is given.
        _reject_unsafe(replaced_expr, expression)

        # Parse the mathematical expression using SymPy
        expr = parse_expr(
            replaced_expr,
            local_dict=local_dict,
            transformations=standard_transformations,
        )

        # Extract all variables from the expression and sort for consistency
        vars = sorted(expr.free_symbols, key=lambda s: s.name)

        #: Compiled NumPy function from SymPy expression
        # Include custom mapping for matmul to numpy.matmul
        self._func = lambdify(
            vars, expr, modules=[{"matmul": np.matmul}, "numpy"]
        )

        #: Ordered list of input port names from expression variables
        self._port_names = [str(var) for var in vars]

        # Create input ports for each variable in the expression
        input_ports = [
            IPort.Configuration(
                name=name,
                type=np.ndarray.__name__,
                timing=Constants.Timing.INHERITED,
            )
            for name in self._port_names
        ]
        input_ports = kwargs.pop(
            Equation.Configuration.Keys.INPUT_PORTS, input_ports
        )

        # Initialize parent IONode with expression and input ports
        super().__init__(
            expression=expression, input_ports=input_ports, **kwargs
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup Equation node and determine output dimensionality.

        Creates pseudo input data based on input context, runs the computation
        to determine output shape, and builds output context with correct
        channel count. This handles dimensionality changes from matrix
        operations.

        Args:
            data: Initial data dictionary for port configuration.
            port_context_in: Input port context with channel counts,
                sampling rates, and frame sizes.

        Returns:
            Output port context with validated configuration and computed
            channel count based on expression output shape.
        """
        # Get reference values from first input port
        first_port = list(port_context_in.keys())[0]
        first_context = port_context_in[first_port]

        frame_size = first_context.get(Constants.Keys.FRAME_SIZE, 1)

        # Create pseudo input data based on input context for each port
        pseudo_data = {}
        for port_name in self._port_names:
            if port_name in port_context_in:
                # Port with context - use its channel count and frame size
                ctx = port_context_in[port_name]
                cc = ctx.get(Constants.Keys.CHANNEL_COUNT, 1)
                fsz = ctx.get(Constants.Keys.FRAME_SIZE, frame_size)
                pseudo_data[port_name] = np.zeros((fsz, cc))
            else:
                # Port without context (e.g., weight matrix passed in data)
                # Use the actual data shape if available
                if port_name in data:
                    pseudo_data[port_name] = data[port_name]
                else:
                    # Fallback: assume scalar
                    pseudo_data[port_name] = np.zeros((1,))

        # Run computation with pseudo data to determine output shape
        pseudo_result = self.step(pseudo_data)
        output_data = pseudo_result[PORT_OUT]

        # Determine output channel count from result shape
        if output_data.ndim == 1:
            # 1D output: each sample produces one value
            output_channel_count = 1
        elif output_data.ndim >= 2:
            # 2D output: (samples, channels)
            output_channel_count = output_data.shape[1]

        # Call parent setup to get base context
        port_context_out = super().setup(data, port_context_in)

        # Override channel count in output context based on computed shape
        for port_name in port_context_out:
            port_context_out[port_name][
                Constants.Keys.CHANNEL_COUNT
            ] = output_channel_count

        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply mathematical expression to input data.

        Evaluates compiled function on current frame of input data in the
        order of sorted variable names from expression.

        Args:
            data: Dictionary with input data arrays for each expression
                variable. Keys are variable names, values are NumPy arrays.

        Returns:
            Dictionary with expression evaluation result on output port.
        """
        # Collect input data in the order expected by the compiled function
        inputs = [data[name] for name in self._port_names]

        # Apply the mathematical function to the input data
        result = self._func(*inputs)

        # Return result in output port format
        return {PORT_OUT: result}

from __future__ import annotations

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


class Equation(IONode):
    """Mathematical expression evaluation node for data transformation.

    Applies custom mathematical expressions to input data using SymPy.
    Automatically creates input ports from expression variables and compiles
    to optimized NumPy functions. Handles 'in' keyword via internal aliasing.
    """

    class Configuration(IONode.Configuration):
        """Configuration class for Equation parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration key constants for the Equation."""

            #: Configuration key for mathematical expression string
            EXPRESSION = "expression"

    def __init__(self, expression: str = None, **kwargs):
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
        replaced_expr = re.sub(r"(\w+)\s*@\s*(\w+)", r"matmul(\1, \2)",
                               replaced_expr)

        # Create symbol mapping for the 'in' keyword alias and matmul function
        local_dict = {
            "__in_alias__": Symbol("in"),
            "matmul": matmul,
        }

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
            port_context_out[port_name][Constants.Keys.CHANNEL_COUNT] = (
                output_channel_count
            )

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

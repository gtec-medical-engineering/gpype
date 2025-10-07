from __future__ import annotations

import re

import numpy as np
from sympy import Symbol, lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode

# Port name constants for convenience
PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


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

        # Create symbol mapping for the 'in' keyword alias
        local_dict = {"__in_alias__": Symbol("in")}

        # Parse the mathematical expression using SymPy
        expr = parse_expr(
            replaced_expr,
            local_dict=local_dict,
            transformations=standard_transformations,
        )

        # Extract all variables from the expression and sort for consistency
        vars = sorted(expr.free_symbols, key=lambda s: s.name)

        # Compile expression to optimized NumPy function
        self._func = lambdify(vars, expr, modules="numpy")

        # Store port names corresponding to expression variables
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
        """Setup Equation node and validate input port configurations.

        Args:
            data: Initial data dictionary for port configuration.
            port_context_in: Input port context with channel counts,
                sampling rates, and frame sizes.

        Returns:
            Output port context with validated configuration.
        """
        return super().setup(data, port_context_in)

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

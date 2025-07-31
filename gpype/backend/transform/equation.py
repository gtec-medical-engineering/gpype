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
    """
    Mathematical expression evaluation node for data transformation.

    The Equation node allows users to apply custom mathematical expressions
    to input data using SymPy symbolic mathematics. It parses string
    expressions, converts them to optimized NumPy functions, and applies
    them to input data arrays in real-time.

    The node automatically creates input ports based on variables found
    in the expression and handles the special case of 'in' as a variable
    name (Python keyword) by using an internal alias.

    Features:
        - Symbolic expression parsing with SymPy
        - Automatic input port generation from expression variables
        - NumPy-optimized function compilation for performance
        - Support for complex mathematical operations
        - Handles Python keyword conflicts (e.g., 'in' variable)

    Note:
        The expression must be a valid SymPy expression. All variables
        in the expression will become input port names.
    """

    class Configuration(IONode.Configuration):
        """Configuration class for Equation parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration key constants for the Equation."""

            EXPRESSION = "expression"

    def __init__(self, expression: str = None, **kwargs):
        """
        Initialize the Equation node with a mathematical expression.

        Parses the provided mathematical expression using SymPy, extracts
        all variables to create input ports, and compiles the expression
        into an optimized NumPy function for real-time evaluation.

        Args:
            expression (str): Mathematical expression as a string. Must be
                a valid SymPy expression. All variables in the expression
                will become input port names. The special variable 'in'
                is handled by internal aliasing.
            **kwargs: Additional configuration parameters passed to IONode.

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
        """
        Set up the Equation node and validate input port configurations.

        Inherits the standard setup behavior from IONode, which validates
        that all required input ports are connected and have compatible
        configurations. No special setup is required for equation evaluation.

        Args:
            data (dict): Initial data dictionary for port configuration.
            port_context_in (dict): Input port context information containing
                channel counts, sampling rates, frame sizes, and data types.

        Returns:
            dict: Output port context with validated configuration.
        """
        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply the mathematical expression to input data.

        Evaluates the compiled mathematical function on the current frame
        of input data. The function expects inputs in the same order as
        the sorted variable names from the original expression.

        Args:
            data (dict): Dictionary containing input data arrays for each
                variable in the expression. Keys are variable names,
                values are NumPy arrays with the current data frame.

        Returns:
            dict: Dictionary containing the result of the expression
                evaluation. The output is stored under the default
                output port name.
        """
        # Collect input data in the order expected by the compiled function
        inputs = [data[name] for name in self._port_names]

        # Apply the mathematical function to the input data
        result = self._func(*inputs)

        # Return result in output port format
        return {PORT_OUT: result}

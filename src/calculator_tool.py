import ast
import operator as op

from langchain_core.tools import tool


@tool
def calculator(query: str) -> str:
    """A safe calculator that evaluates math expressions."""

    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
        ast.FloorDiv: op.floordiv,
        ast.USub: op.neg,
    }

    def eval_expr(expr):
        def _eval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return allowed_ops[type(node.op)](_eval(node.operand))
            else:
                raise TypeError(f"Unsupported type: {type(node)}")

        node = ast.parse(expr, mode="eval").body
        return _eval(node)

    try:
        result = eval_expr(query)
        return str(result)
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Error: {e}"

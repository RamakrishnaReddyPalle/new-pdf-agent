# Safe eval (numexpr/sympy), units & validation
# packages/calculators/executor.py
from __future__ import annotations
import math
from typing import Any, Dict, Optional

# Optional dependencies
_HAS_SYMPY = False
try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

_HAS_PINT = False
try:
    import pint
    _HAS_PINT = True
except Exception:
    _HAS_PINT = False

from .parser import ToolSpec

class CalcError(ValueError):
    pass

def _mk_unit_registry():
    if not _HAS_PINT:
        return None
    try:
        return pint.UnitRegistry()
    except Exception:
        return None

def evaluate(spec: ToolSpec, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a ToolSpec with provided inputs.
    - If spec.target is set and SymPy is available, solve for target if needed.
    - Otherwise, subs into expression and evaluate.
    - If Pint is available and inputs include units (e.g., {'m': '5 meter'}), we try to carry units.
    Returns: {'ok': bool, 'value': any, 'unit': str|None, 'detail': str|None}
    """
    expr = spec.expression or ""
    if "=" in expr:
        lhs, rhs = [s.strip() for s in expr.split("=", 1)]
    else:
        lhs, rhs = None, expr.strip()

    # Prepare symbols
    if not _HAS_SYMPY:
        # very light fallback: Python eval on sanitized namespace
        # WARNING: no units here; trusted inputs only (this is dev fallback).
        ns = {k: float(str(v).split()[0]) for k, v in inputs.items() if isinstance(v, (int, float, str))}
        try:
            val = eval(rhs, {"__builtins__": {}}, ns)
            return {"ok": True, "value": float(val), "unit": None, "detail": "python-eval fallback"}
        except Exception as e:
            raise CalcError(f"Evaluation failed without sympy: {e}")

    # With SymPy (and optionally Pint)
    sym_vars = {}
    for v in spec.inputs:
        sym_vars[v.name] = sp.symbols(v.name)

    # Unit handling with Pint
    ureg = _mk_unit_registry() if _HAS_PINT else None
    Q_ = ureg.Quantity if ureg else None

    # Parse numeric inputs (possibly with units)
    subs_map = {}
    for name, val in inputs.items():
        if isinstance(val, (int, float)):
            subs_map[name] = float(val)
        elif isinstance(val, str) and ureg:
            try:
                qty = ureg(val)
                subs_map[name] = qty
            except Exception:
                # try plain float at least
                try:
                    subs_map[name] = float(val)
                except Exception:
                    pass
        else:
            try:
                subs_map[name] = float(val)
            except Exception:
                pass

    # Build sympy expression
    try:
        expr_sym = sp.sympify(rhs, locals=sym_vars)
    except Exception as e:
        raise CalcError(f"Could not parse expression '{rhs}': {e}")

    # Evaluate
    # Case 1: we solve for target variable if lhs exists and target is among symbols
    if spec.target and lhs:
        target_sym = sym_vars.get(spec.target)
        # If the expression was 'lhs = rhs' and target == lhs, we can directly subs/eval rhs
        if spec.target == lhs:
            val = _eval_with_units(expr_sym, subs_map, ureg)
            return val
        else:
            # Solve the equation for target (lhs - rhs = 0)
            eq = sp.Eq(sym_vars.get(lhs, sp.symbols(lhs)), expr_sym)
            try:
                sol = sp.solve(eq, target_sym, dict=True)
            except Exception as e:
                raise CalcError(f"Sympy solve failed: {e}")
            if not sol:
                raise CalcError("No solution found for target.")
            # pick first solution
            solution = sol[0][target_sym]
            return _eval_with_units(solution, subs_map, ureg)
    else:
        # Just evaluate rhs with subs
        return _eval_with_units(expr_sym, subs_map, ureg)

def _eval_with_units(sym_expr, subs_map, ureg) -> Dict[str, Any]:
    """Evaluate a sympy expression possibly containing Pint quantities."""
    # Substitute numeric values (including Pint quantities)
    # For Pint, we cannot directly subs quantities into SymPy symbols.
    # Workaround: evaluate symbol-by-symbol in Python if quantities present.
    has_qty = any(hasattr(v, "units") for v in subs_map.values())

    if not has_qty:
        # Pure numeric path
        val = sym_expr.evalf(subs=subs_map)
        try:
            val = float(val)
        except Exception:
            pass
        return {"ok": True, "value": val, "unit": None, "detail": "sympy-numeric"}

    # Quantity path: turn SymPy expression into a lambda and evaluate in Pint world
    # WARNING: we allow only a small set of safe functions
    import math
    safe = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "exp": math.exp, "sqrt": math.sqrt,
        "pi": math.pi,
    }
    # Build a python function from symbols
    syms = sorted({str(s) for s in sym_expr.free_symbols})
    f = sp.lambdify(tuple(sp.symbols(syms)), sym_expr, modules=[safe])

    # Extract args in symbol order
    args = []
    for s in syms:
        v = subs_map.get(s)
        if v is None:
            raise CalcError(f"Missing input: {s}")
        args.append(v)

    out = f(*args)
    if ureg and hasattr(out, "units"):
        try:
            # return magnitude + unit string
            return {"ok": True, "value": out.magnitude, "unit": f"{out.units:~P}", "detail": "pint-quantity"}
        except Exception:
            return {"ok": True, "value": out, "unit": None, "detail": "pint-raw"}
    return {"ok": True, "value": out, "unit": None, "detail": "python-lambda"}

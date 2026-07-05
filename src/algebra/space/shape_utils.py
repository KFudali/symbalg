from __future__ import annotations

from typing import Any
import numpy as np

from tools.symbolic.optype import BinaryOpType, MatOpType
from .fieldshaped import FieldShape, FieldShaped


def pick_component(
    source: FieldShape, comp: int | tuple[int | slice, ...]
) -> tuple[int | slice, ...]:
    if isinstance(comp, int):
        comps = source.components[0]
        if comp >= comps:
            raise ValueError(
                f"Trying to pick component {comp} of field with {comps} components"
            )
        return (comp, slice(None, None))
    if isinstance(comp, tuple):
        if len(source.components) != len(comp):
            raise ValueError(
                (
                    "Can only pick component with int argument if the source field ",
                    "component count matches query length. souce components ",
                    f"{source.components}, aksed for component: {comp}",
                )
            )
        if all(isinstance(c, (int, slice)) for c in comp):
            return (*comp, slice(None, None))
    raise ValueError(
        "Can only pick component using int | tuple[int, ...] | tuple[slice, ...]"
    )


def project_shape(left: FieldShape, right: FieldShape, optype: MatOpType) -> FieldShape:
    """Returns resulting shape of optype operation betweend fieldshapes right, left"""
    left_comps = left.components
    right_comps = right.components

    if optype == MatOpType.DOT:
        if not left_comps or not right_comps:
            return FieldShape(left.space, left_comps or right_comps)
        return FieldShape(left.space, left_comps[:-1] + right_comps[1:])

    if optype == MatOpType.INNER:
        if not left_comps or not right_comps:
            return FieldShape(left.space, left_comps or right_comps)
        return FieldShape(left.space, ())

    if optype == MatOpType.OUTER:
        return FieldShape(left.space, left_comps + right_comps)
    return NotImplemented


def project_einsum(left: FieldShape, right: FieldShape, optype: MatOpType) -> str:
    """Returns str subscripts that need to be passed to np.einsum to perform optype
    operation between left and right fieldshapes"""
    left_comps = left.components
    right_comps = right.components
    letters = "abcdefghijklmnopqrstuvwxyz"
    n, m = len(left_comps), len(right_comps)

    if optype == MatOpType.DOT:
        if not left_comps or not right_comps:
            return "...,...->..."
        a_labels = letters[:n]
        b_labels = letters[n - 1 : n - 1 + m]
        result_labels = a_labels[:-1] + b_labels[1:]
        return f"{a_labels}...,{b_labels}...->{result_labels}..."

    if optype == MatOpType.INNER:
        if not left_comps or not right_comps:
            return "...,...->..."
        a_labels = letters[:n]
        b_labels = letters[:n]
        return f"{a_labels}...,{b_labels}...->..."

    if optype == MatOpType.OUTER:
        if not left_comps or not right_comps:
            return "...,...->..."
        a_labels = letters[:n]
        b_labels = letters[n : n + m]
        result_labels = a_labels + b_labels
        return f"{a_labels}...,{b_labels}...->{result_labels}..."

    return "...,...->..."

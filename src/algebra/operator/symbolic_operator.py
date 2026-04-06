from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Self, Any
from .operator import Operator
from tools.symbolic import Symbolic, SymbolicNode, Value, UnaryOp, BinaryOp, Op
from algebra.exceptions import ShapeMismatchError


@dataclass(frozen=True)
class OperatorValue(Value[Operator]):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.value.input_shape
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.value.output_shape

@dataclass(frozen=True)
class OperatorUnaryOp(UnaryOp[Operator]):
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

@dataclass(frozen=True)
class OperatorBinaryOp(BinaryOp[Operator]):
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

OperatorNode = SymbolicNode[Operator]

class SymbolicOperator(Operator, Symbolic[Operator]):
    COMPATIBLE_TYPES = ( 
        Operator,
        OperatorBinaryOp, 
        OperatorUnaryOp, 
        OperatorValue,
        float, np.ndarray
    )
   
    def __init__(self, operator: OperatorNode):
        Symbolic.__init__(self, operator)
        Operator.__init__(self, operator.input_shape, operator.output_shape)

    def copy(self) -> Self: pass

    def _new(self, op: Op[Operator]):
        return SymbolicOperator(op)

    def _make_value(self, operator: Operator):
        return OperatorValue(operator)

    def _make_unary(self, optype: UnaryOp.OpType) -> OperatorUnaryOp:
        return OperatorUnaryOp(
            optype, self.base_op, self.input_shape, self.output_shape
        )
    
    def _make_binary(self, other: OperatorNode, optype: BinaryOp.OpType):
        return OperatorBinaryOp(
            optype, self.base_op, self._wrap(other), 
            self.input_shape, self.output_shape
        )

    def _assert_compatible(self, other: Any):
        compatible_types = (SymbolicOperator, * SymbolicOperator.COMPATIBLE_TYPES)
        if isinstance(other, float): return
        if isinstance(other, np.ndarray):
            if other.shape != self.output_shape:
                raise ShapeMismatchError((
                    "Can only combine operator with array that matches its output_shape. ",
                    f"self.output_shape: {self.output_shape}, array.shape: {other.shape}."
                ))
        if isinstance(other, compatible_types):
            if self.input_shape != other.input_shape:
                raise ShapeMismatchError((
                    "Can only combine OperatorNodes of equal input_shape.",
                    f"self.input_shape: {self.input_shape},",
                    f"other.input_shape: {other.input_shape}."
                ))
            if self.output_shape != other.output_shape:
                raise ShapeMismatchError((
                    "Can only combine OperatorNodes of equal output_shape.",
                    f"self.output_shape: {self.output_shape},",
                    f"other.output_shape: {other.output_shape}."
                ))
            raise ValueError((
                "SymbolicOperator can only be combined with objects of type: ",
                f"{compatible_types}."
            ))
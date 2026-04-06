

# SymbolicExpression() -> chains only expressions and floats/np.arrays also, output_shape is const and is of self._base shape.
# Since Expression does not implement add sub mul div neg it should upon folding call eval() and fold this way
#     resolve() -> Expression()
#     eval() -> np.ndarray (calls self.resolve().eval())
# SymbolicOperator() -> chains only operators calls their magics, also computes input/output shape on the way
#     resolve() -> Operator()
#     apply(input, output) (calls self.resolve().apply(input, output))

class SymbolicOperator(): pass


# class SymbolicExpression():

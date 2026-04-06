Dx.laplace(field) | FieldOperatorExpression
Dt.euler(field) | FieldOperatorExpression

Dx + Dt -> Symbolic

resolve()
FieldOperatorExpression + FieldOperatorExpression -> FieldOperatorExpression

Dx.laplace(field) + Dt.euler(other_field) -> FieldOperatorExpression:
resolve()
FieldOperatorExpression + FieldOperatorExpression | different fields -> FieldExpression

FieldOperatorExpresion(OperatorWrapper)
    def __init__(self, field: Field, operator: Operator, lifting: Expression = None):


FieldOperatorExpression() | SymbolicExpression
    eval() -> np.ndarray
    apply(self, field, out) -> np.ndarray
    operator -> Operator

Expression 
    eval() -> float / array
Operator
    apply(input, output)

Symbolic() -> chains whatever class exposes some API for what to do on resolving tree
    resolve() -> Any
SymbolicExpression() -> chains only expressions and floats/np.arrays also, output_shape is const
    resolve() -> Expression()
    eval() -> np.ndarray (calls self.resolve().eval())
SymbolicOperator() -> chains only operators calls their magics, also computes input/output shape on the way
    resolve() -> Operator()
    apply(input, output) (calls self.resolve().apply(input, output))

EulerDt(FieldOperatorExpression):
    def __init__(self, field: Field):
        output_shape = operator.output_shape
        operator = StencilOperaotr.ones()
        lifting = field.past().value() / field.space.time.dt()
        symbolic_op = SymbolicOperator(operator)
        symbolic_op /= field.space.time.dt() | SymbolicOperator .resolve() -> Operator
        super().__init__(field, symbolic_op, lifting)

    def value(self) -> SymbolicExpression:
        return SymbolicExpression(CallableExpression(self.eval, self.output_shape))

    @property
    def operator() -> SymbolicOperator(): pass

    @property
    def lifting() -> SymbolicExpression(): pass

    @property
    def take_lifting() -> SymbolicExpression(): pass

    def eval() -> np.ndarray:
        pass

    def __add__(self, Operator, Expression, FieldOperatorExpression):
        if isinstance(FieldOperatorExpression):
            if self.field == other.field:
                return FieldOperatorExpression(
                    self.field, self.operator + other.operator
                )
            else:
                return self.value() + other.value()
        if isinstance(other: float | Expression):
            return FieldOperatorExpression(
                self.field, self.operator, self.lifting + other
            )

LESystem(System)
    def __init__(self, lhs: Operator, rhs: Expression | None):
        self._lhs = lhs
        slef._rhs = rhs

    def sovle(self) -> SymbolicExpression():


    def _assemble(self) -> tuple[LinearOperator, np.ndarray]
        if isinstance(lhs, FieldOperatorExpression):
            rhs = rhs + lhs.lifting()
            lhs = lhs.operator()
        ...

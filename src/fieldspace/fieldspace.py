SpaceObject():
    shape()

Field()

DiscreteSpace(Space)
    domain -> Domain
    time -> TimeSeries

    operators ->
        .dx -> FieldOperatorExpression
        .dt -> FieldOperatorExpression

    shape ->

fieldspace
    core/
        field/
            abstract_field..
        space/
        operators/
    field.py
    space.py

space/core
    

core/

import fieldspace
from fieldspace.operators import Dx, Dt
import domain

grid = StructuredGrid(20,20)
fd_domain = domain.fd.FDDomain(grid)
space = fieldspace.FieldSpace(fd_domain)

field = space.fields.scalar()
Dx.laplace(field)
Dt.expl.euler(field)

fieldspace.field.scalar



Space(Generic[Domain])
    def __init__(domain: Domain) 
    shape
    time
    operators





Dx(field):
    return field.space.discrete.operators.laplace(field)

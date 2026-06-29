from algebra.space import FieldShape


def pick_component(
    source: FieldShape, comp: int | tuple[int, ...] | tuple[slice, ...]
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
        if all(isinstance(c, int) for c in comp):
            return (*comp, slice(None, None))
        if all(isinstance(c, slice) for c in comp):
            return (*comp, slice(None, None))
    raise ValueError(
        "Can only pick component using int | tuple[int, ...] | tuple[slice, ...]"
    )

from numpy import ndarray


# TODO: Collapse can mean that
#  1) One of the solutions is not available anymore
#  2) An entire regime is not available anymore
#  .
#  These collapses can always come from any of the edges/borders of the solution spaces,
#  e.g., you cannot provide sufficient resources anymore. As a consequence, the shape
#  might not be convex anymore, but this should not be a problem with the sampling
#  mechanisms I'm currently using

# TODO: Supposed that this can be a highly-dimensional array, we use one dimension and multiply it
#  from one axis (likely the higher one) with a constant that is declining over the rows.

def collapse_offer(fitness_table: ndarray) -> ndarray:
        pass

class CollapseEngine:

    def __init__(self):
        pass


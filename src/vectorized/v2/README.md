This version uses query-level parallelization with the stack and flattening
of v1. Since the queries are not spatially ordered, the result is a computational
mess, where the time to completion is much higher than before.

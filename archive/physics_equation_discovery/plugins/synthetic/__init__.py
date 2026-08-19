"""A synthetic regression domain, with a known generating formula.

Its value is that it is not a science. The engine claims to be domain
independent, and the way that claim gets tested is by driving it with a second
domain sharing no code path with the first: if the audit runs here unmodified,
the seam is real rather than asserted.

It also gives the benchmark scripts a fast, dependency-free dataset, which is
why `data.py` is imported well beyond this plugin.
"""

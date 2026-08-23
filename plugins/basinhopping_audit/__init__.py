"""Audits `scipy.optimize.basinhopping` against budget-matched restarts of its
own local minimizer. See SPEC.md (issue #15) for scope.

Not registered with `engine.registry.PluginRegistry`: this is a self-
contained research script over an existing library, not a new domain the
rest of the platform needs to discover (no `AuditProblemSource` -- the three
test functions in `functions.PROBLEM_SET` are looked up directly).
"""

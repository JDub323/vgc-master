"""BERMUDA: Bermudan-Exercise Regression Monte Carlo for Doubles play.

See plan.md at the repo root for the architecture. Nothing in this package
depends on the tree-search / policy-network stack; only sim plumbing
(env.py), the log tracker (data.py), the action space (actions.py), and the
belief particle filter (as a scenario materializer) are reused.
"""

"""evalkit — a local, record-keeping eval framework for conversational agents.

Built on Inspect AI (logs, viewer, scoring, epochs). It owns the parts that decide whether
a number means anything: suites and templating, the simulated user, the scorers, run
provenance, and the paired statistics that separate a real improvement from noise.

It does not know what it is evaluating, and nothing in it names a product. The agent under
test sits behind a **target** (:mod:`evalkit.target`), resolved at runtime from
``EVAL_TARGET``.
"""

__version__ = "0.2.0"

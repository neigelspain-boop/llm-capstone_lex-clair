"""Investigation passes.

Each module exposes `PASS_NAME` and `run(ctx) -> PassResult`. Plain functions,
hard-wired by name in the orchestrator — no base class, no registry, no
discovery. The contract is enforced by one constructor (`store.finding`, which
takes `pass_name` as a required positional) and one test asserting every
finding a module emits carries that module's own name.
"""

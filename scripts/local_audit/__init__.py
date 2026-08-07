"""Local-only code-audit harness for lex-clair, driven by a local Ollama
model (qwen3:14b). Report-only: every pass produces structured findings in
audit/findings.jsonl and a regenerated audit/DIGEST.md — nothing here ever
writes to a source file outside audit/. See README.md in this directory.
"""

"""LLM output evaluation harness (rubric: needs >=2 approaches compared,
best one wins) — e.g. different prompts, models, or providers, judged by
an LLM-as-judge or a simple heuristic against reference answers.

Run with:
    uv run python -m eval.llm_eval
"""
from typing import Callable

from src.rag.llm_client import call_llm
from src.rag.prompts import build_prompt

# TODO: replace with a real labelled set: [{"question": ..., "contexts": [...], "reference": ...}, ...]
EVAL_SET: list[dict] = []


def judge(question: str, generated: str, reference: str) -> float:
    # TODO: LLM-as-judge call, or a simpler similarity metric
    raise NotImplementedError


def approach_claude(item: dict) -> str:
    prompt = build_prompt(item["question"], item["contexts"])
    return call_llm(prompt, provider="claude")


def approach_openai(item: dict) -> str:
    prompt = build_prompt(item["question"], item["contexts"])
    return call_llm(prompt, provider="openai")


APPROACHES: dict[str, Callable[[dict], str]] = {
    "claude": approach_claude,
    "openai": approach_openai,
}


def main():
    for name, approach in APPROACHES.items():
        scores = []
        for item in EVAL_SET:
            generated = approach(item)
            scores.append(judge(item["question"], generated, item["reference"]))
        avg = sum(scores) / len(scores) if scores else 0
        print(name, {"avg_score": avg})


if __name__ == "__main__":
    main()

"""
agent reliability benchmark w/ Respan
======================================
stress-tests a RAG agent under adversarial retrieval conditions using Respan
for full agent trace observability.

motivation: context length alone doesn't break modern 128k models — adversarial
retrieval does. this benchmark injects conflicting spec versions (distractors)
that contain plausible but wrong answers, forcing the agent to adjudicate
between versions rather than just retrieve. the two relevant sections of the
canonical spec are maximally separated, requiring true long-range synthesis
under noise.

stress variable: distractor density (# of conflicting spec versions injected)
not raw token count.

setup:
    pip install respan-ai openai tabulate matplotlib python-dotenv

usage:
    set RESPAN_API_KEY, OPENAI_API_KEY in .env
    python benchmark.py

output:
    - console table: quality scores per model x distractor level
    - degradation_curve.png: quality vs distractor density plot
    - all agent traces logged to platform.respan.ai => see dash
"""

import os
import json
import time
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
from openai import OpenAI, RateLimitError
from respan import Respan, workflow, task
from dotenv import load_dotenv

load_dotenv()

# init respan - auto-instr openai
Respan()

# load client
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODELS = ["gpt-4o", "gpt-4o-mini"]

# stress variable: how many conflicting spec versions are injected
# 0 = clean retrieval, 3 = maximum adversarial noise
DISTRACTOR_LEVELS = [
    ("0_distractors", 0),
    ("1_distractor",  1),
    ("2_distractors", 2),
    ("3_distractors", 3),
]

# canonical spec (v2.3) 
# split into two halves — injected at opposite ends of context to force
# long-range synthesis. distractors are buried between them.

SPEC_V23_TOP = """
=== Orion Inference Platform — Spec v2.3 (CANONICAL) — Part 1 of 2 ===

[SECTION A — Request Router]
The request router assigns incoming inference requests to one of three execution
tiers based on payload size and SLA class. Tier-1 handles requests under 512
tokens with a p99 latency target of 80ms. Tier-2 handles 512–4096 tokens with
a 400ms target. Tier-3 handles anything above 4096 tokens and has no hard
latency SLA but enforces a 30s timeout. If the router cannot determine SLA
class from the request header, it defaults to Tier-2. Priority requests bypass
tier assignment and are routed directly to the GPU-B pool regardless of size.

[SECTION B — Cache Layer]
The cache layer sits between the router and model workers. It uses a two-level
structure: an in-process LRU cache (L1, 256MB per worker) and a shared Redis
cluster (L2, 48GB total). L1 hit latency is under 1ms. L2 hit latency is
8–12ms. On a cache miss, the response is written to both L1 and L2 with a TTL
of 300 seconds. Cache keys are SHA-256 hashes of the normalized prompt.
Tier-3 requests bypass the cache entirely due to key size constraints.
"""

SPEC_V23_BOTTOM = """
=== Orion Inference Platform — Spec v2.3 (CANONICAL) — Part 2 of 2 ===

[SECTION C — Model Workers]
Three worker pools run in parallel: a CPU pool (16 workers, handles Tier-1
overflow only), a GPU-A pool (8 workers on A100s, handles Tier-1 and Tier-2),
and a GPU-B pool (4 workers on H100s, reserved for Tier-3 and priority
requests). Workers emit heartbeat signals every 5 seconds. A worker missing
two consecutive heartbeats is marked unhealthy and removed from the pool.
GPU-B workers run a quantized model variant to reduce memory pressure.

[SECTION D — Failure Modes]
If all GPU-A workers are saturated, Tier-2 requests spill into the CPU pool
with a degraded latency SLA of 1200ms. If the Redis cluster is unreachable,
L2 cache is bypassed and only L1 is used. A full GPU-B pool failure causes
Tier-3 requests to be rejected with a 503 — there is no automatic fallback
from GPU-B to GPU-A for Tier-3 requests. Priority requests share the GPU-B
pool with Tier-3; a full GPU-B pool rejects both with a 503.
"""

# distractor specs 
# each version has subtly wrong numbers that look authoritative.
# v2.1 and v2.2 are old — v2.3-draft was never approved.
# models that blend versions or miss the canonical label will get wrong answers.

DISTRACTOR_V21 = """
=== Orion Inference Platform — Spec v2.1 (SUPERSEDED) ===

[SECTION A — Request Router]
Tier-1 handles requests under 256 tokens with a p99 latency target of 100ms.
Tier-2 handles 256–2048 tokens with a 600ms target. Tier-3 handles anything
above 2048 tokens with a 60s timeout. Priority requests route to GPU-A pool.
Default tier on missing SLA class: Tier-1.

[SECTION B — Cache Layer]
L1 cache: 128MB per worker. L2 Redis cluster: 24GB total. TTL: 600 seconds.
Cache keys are MD5 hashes of the raw prompt. All tiers use the cache.

[SECTION C — Model Workers]
CPU pool: 8 workers. GPU-A pool: 12 workers on V100s. GPU-B pool: 2 workers
on A100s, reserved for Tier-3 only. Heartbeat interval: 10 seconds.
A worker missing three consecutive heartbeats is marked unhealthy.

[SECTION D — Failure Modes]
GPU-A saturation causes Tier-2 spill to CPU pool at 2000ms degraded SLA.
Redis failure causes full cache bypass (no L1 fallback). GPU-B failure
causes Tier-3 to fall back to GPU-A with a 5000ms SLA.
"""

DISTRACTOR_V22 = """
=== Orion Inference Platform — Spec v2.2 (SUPERSEDED) ===

[SECTION A — Request Router]
Tier-1 handles requests under 1024 tokens with a p99 latency target of 120ms.
Tier-2 handles 1024–8192 tokens with a 500ms target. Tier-3 handles anything
above 8192 tokens with a 45s timeout. Priority requests route to GPU-A pool
with elevated queue priority. Default tier on missing SLA class: Tier-3.

[SECTION B — Cache Layer]
L1 cache: 512MB per worker. L2 Redis cluster: 96GB total. TTL: 150 seconds.
Cache keys are SHA-1 hashes of the normalized prompt. Tier-3 requests use
L2 only (L1 bypassed due to size). 

[SECTION C — Model Workers]
CPU pool: 32 workers. GPU-A pool: 6 workers on A100s. GPU-B pool: 8 workers
on H100s, handles Tier-2, Tier-3, and priority. Heartbeat interval: 3 seconds.
A worker missing one heartbeat is marked unhealthy.

[SECTION D — Failure Modes]
GPU-A saturation causes Tier-2 rejection with 429. Redis failure causes
requests to be held in a retry queue for up to 30 seconds. GPU-B failure
causes Tier-3 and priority requests to fall back to GPU-A automatically.
"""

DISTRACTOR_V23_DRAFT = """
=== Orion Inference Platform — Spec v2.3-draft (NOT APPROVED — DO NOT USE) ===

[SECTION A — Request Router]
Tier-1 handles requests under 512 tokens with a p99 latency target of 50ms.
Tier-2 handles 512–4096 tokens with a 200ms target. Tier-3 handles anything
above 4096 tokens with a 20s timeout. Priority requests route to a dedicated
priority pool (GPU-C, not yet provisioned). Default tier: Tier-2.

[SECTION B — Cache Layer]
L1 cache: 256MB per worker. L2 Redis cluster: 48GB total. TTL: 60 seconds.
Cache keys are SHA-256 hashes. Tier-3 requests use L2 only.

[SECTION C — Model Workers]
CPU pool: 16 workers. GPU-A pool: 8 workers. GPU-B pool: 4 workers, handles
Tier-3 only (priority requests moved to GPU-C in this version). Heartbeat: 5s.
Two missed heartbeats => unhealthy.

[SECTION D — Failure Modes]
GPU-A saturation: Tier-2 spills to CPU at 800ms degraded SLA. Redis failure:
L1 only. GPU-B failure: Tier-3 rejected with 503, no fallback.
Priority pool failure: priority requests fall back to GPU-B.
"""

ALL_DISTRACTORS = [DISTRACTOR_V21, DISTRACTOR_V22, DISTRACTOR_V23_DRAFT]

# multi-hop QA pairs 
# every answer requires:
# 1. identifying v2.3 as canonical (ignoring distractors)
# 2. connecting section A + section C or B + D (which are maximally separated)
# wrong answers from distractors are specific and plausible — not obviously wrong

QA_PAIRS = [
    {
        "question": "A 600-token priority request arrives. Which worker pool handles it and what is the p99 latency target for its size tier?",
        "answer": "Priority requests bypass tier assignment and go directly to GPU-B (H100s) per v2.3 Section A. The request is 600 tokens which falls in Tier-2 (512-4096 tokens), which has a 400ms p99 latency target — though priority routing overrides normal tier assignment.",
    },
    {
        "question": "The Redis cluster goes down. A 300-token request arrives and misses L1 cache. What happens and which worker pool serves it?",
        "answer": "Per v2.3: Redis down means L2 is bypassed, only L1 is available. The request already missed L1, so it's a full cache miss and goes to a model worker. 300 tokens is Tier-1, so it routes to the GPU-A pool (or CPU pool if GPU-A is saturated). No cache write occurs since Redis is down.",
    },
    {
        "question": "All GPU-A workers are saturated and a Tier-2 request arrives. What is the degraded latency SLA and what happens if GPU-B is also full?",
        "answer": "Per v2.3 Section D: Tier-2 spills to the CPU pool with a degraded SLA of 1200ms. GPU-B failure only affects Tier-3 and priority requests — a Tier-2 spillover to CPU is unaffected by GPU-B status.",
    },
    {
        "question": "How many GPU-B workers exist and what request types do they handle? What happens to those request types if all GPU-B workers fail?",
        "answer": "Per v2.3 Section C: 4 GPU-B workers on H100s, reserved for Tier-3 and priority requests. Per Section D: if GPU-B fully fails, Tier-3 requests and priority requests are rejected with a 503 — there is no automatic fallback to GPU-A for either type.",
    },
]

# filler noise 
# injected around distractors to make them feel like realistic retrieved chunks

FILLER_CHUNKS = [
    "Deployment log [2026-03-14]: Orion platform v2.3 rolled out to prod-us-east-1. Rollback window: 24h. On-call: @sre-team.",
    "Slack thread [#infra-alerts]: Redis cluster latency spike detected at 14:32 UTC. P99 hit 45ms. Auto-remediation triggered. Resolved 14:38.",
    "Runbook excerpt [GPU worker recovery]: SSH to affected worker node. Run `systemctl restart orion-worker`. Verify heartbeat resumes within 30s.",
    "PR description [#4821]: Refactor tier assignment logic to support future Tier-4 (streaming). No behavior change for existing tiers in this PR.",
    "Postmortem [2026-02-01]: GPU-B pool exhaustion caused 23-minute Tier-3 outage. Root cause: batch job misconfigured as priority request. Fix: added priority flag validation at router.",
    "Meeting notes [platform sync 2026-03-10]: Agreed to deprecate v2.1 spec. v2.2 already archived. v2.3 is the single source of truth. v2.3-draft proposals tabled pending GPU-C procurement.",
    "Monitoring alert [resolved]: Tier-1 p99 exceeded SLA threshold (>80ms) for 3 minutes. Cause: L1 cache eviction storm on worker pool restart. No user impact beyond latency.",
    "Wiki comment: Note — v2.1 and v2.2 specs are kept in this repo for historical reference only. Do not use for operational decisions.",
]

def generate_context(distractor_count: int) -> str:
    """
    build the full context:
    - v2.3 part 1 (top)
    - filler noise
    - distractor specs (buried in middle)
    - more filler noise  
    - v2.3 part 2 (bottom)
    maximally separates the two canonical spec halves.
    """
    distractors = random.sample(ALL_DISTRACTORS, distractor_count)

    # pad with filler noise so distractors don't feel isolated
    filler_top    = "\n\n".join(random.choices(FILLER_CHUNKS, k=4))
    filler_middle = "\n\n".join(random.choices(FILLER_CHUNKS, k=4))
    filler_bottom = "\n\n".join(random.choices(FILLER_CHUNKS, k=4))

    middle_sections = [filler_top]
    for d in distractors:
        middle_sections.append(d)
        middle_sections.append("\n\n".join(random.choices(FILLER_CHUNKS, k=2)))
    middle_sections.append(filler_middle)

    return "\n\n".join([
        SPEC_V23_TOP,
        "\n\n".join(middle_sections),
        filler_bottom,
        SPEC_V23_BOTTOM,
    ])


# retry w/ exponential backoff for rate limits
def call_with_retry(fn, retries=5, base_wait=10):
    for attempt in range(retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"         ⏳ rate limit, waiting {wait}s...")
            time.sleep(wait)


# agent tasks
@task(name="retrieve_context")
def retrieve_context(distractor_count: int) -> str:
    """
    build adversarial context:
    - canonical v2.3 spec split and maximally separated
    - distractor_count conflicting old versions buried between
    - filler noise throughout
    """
    return generate_context(distractor_count)


@task(name="generate_answer")
def generate_answer(question: str, context: str, model: str) -> tuple[str, dict]:
    """call LLM with adversarial context. must identify canonical version + multi-hop."""
    system = (
        "You are a precise question-answering assistant for an internal engineering platform. "
        "The context may contain multiple versions of the same spec. "
        "Always use the CANONICAL (v2.3) version — ignore superseded or draft versions. "
        "Answer using ONLY information from the canonical spec. Be concise — two to three sentences max."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    response = call_with_retry(lambda: openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=250,
    ))
    answer = response.choices[0].message.content.strip()
    usage = {
        "input_tokens":  response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens":  response.usage.total_tokens,
    }
    return answer, usage


# evals done w/ llm as a judge w/ qa dataset and goldens
@task(name="evaluate_answer")
def evaluate_answer(question: str, predicted: str, expected: str) -> dict:
    """LLM-as-judge: strict scoring — penalizes distractor blending heavily."""
    system = """You are a strict answer quality judge for an engineering QA system.

        The context contained multiple conflicting spec versions. The correct answer
        uses ONLY the canonical v2.3 spec. Penalize heavily if the predicted answer
        uses numbers or behaviors from superseded versions (v2.1, v2.2, v2.3-draft).

        Score on two dimensions (0.0 to 1.0):
        - faithfulness: does the answer correctly use v2.3 facts and connect multiple sections? penalize wrong numbers, wrong pool assignments, wrong failure behaviors, or distractor blending.
        - conciseness: is the answer appropriately brief without padding?

        Return ONLY a JSON object — no markdown, no explanation:
        {"faithfulness": <float>, "conciseness": <float>, "overall": <float>, "notes": "<one sentence>"}

        overall = 0.8 * faithfulness + 0.2 * conciseness"""

    user = f"Question: {question}\nExpected (v2.3 canonical): {expected}\nPredicted: {predicted}"

    response = call_with_retry(lambda: openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=150,
    ))
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": 0.0, "conciseness": 0.0, "overall": 0.0, "notes": "judge parse error"}


# toplevel
@workflow(name="rag_agent_run")
def run_rag_agent(question: str, expected: str, model: str, distractor_label: str, distractor_count: int) -> dict:
    """
    full RAG agent run: retrieve -> generate -> evaluate.
    respan traces this as a workflow with nested task spans.
    """
    context = retrieve_context(distractor_count)
    answer, usage = generate_answer(question, context, model)
    scores = evaluate_answer(question, answer, expected)

    return {
        "model": model,
        "distractor_level": distractor_label,
        "distractor_count": distractor_count,
        "total_tokens": usage["total_tokens"],
        "answer": answer,
        "scores": scores,
    }


def main():
    random.seed(42)

    print(f"\n{'='*65}")
    print("  Agent Reliability Benchmark — Respan")
    print(f"  stress mode: adversarial retrieval (conflicting spec versions)")
    print(f"  {len(QA_PAIRS)} questions x {len(DISTRACTOR_LEVELS)} distractor levels x {len(MODELS)} models")
    print(f"  all traces -> platform.respan.ai")
    print(f"{'='*65}\n")

    # results[model][distractor_label] = list of overall scores
    results = {m: {label: [] for label, _ in DISTRACTOR_LEVELS} for m in MODELS}

    total_runs = len(QA_PAIRS) * len(DISTRACTOR_LEVELS) * len(MODELS)
    run = 0

    for label, distractor_count in DISTRACTOR_LEVELS:
        for model in MODELS:
            for qa in QA_PAIRS:
                run += 1
                print(f"[{run:>3}/{total_runs}] {model:<14} distractors={label:<15} Q: \"{qa['question'][:42]}...\"")

                result = run_rag_agent(
                    question=qa["question"],
                    expected=qa["answer"],
                    model=model,
                    distractor_label=label,
                    distractor_count=distractor_count,
                )
                score = result["scores"].get("overall", 0)
                results[model][label].append(score)

                print(f"         → overall={score:.2f}  tokens={result['total_tokens']}  note: {result['scores'].get('notes', '')[:60]}")
                time.sleep(0.3)
            print()

    # summ table
    rows = []
    for model in MODELS:
        row = [model]
        baseline_score = None
        for label, _ in DISTRACTOR_LEVELS:
            scores = results[model][label]
            avg = sum(scores) / len(scores)
            if baseline_score is None:
                baseline_score = avg
                row.append(f"{avg:.3f}")
            else:
                delta = avg - baseline_score
                sign = "+" if delta >= 0 else ""
                row.append(f"{avg:.3f} ({sign}{delta:.3f})")
        rows.append(row)

    headers = ["Model"] + [label for label, _ in DISTRACTOR_LEVELS]
    print(f"\n{'='*65}")
    print("  Quality Score (overall) by Distractor Level")
    print(f"  Delta shown relative to 0-distractor baseline")
    print(f"{'='*65}")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # degr curve plot
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [label for label, _ in DISTRACTOR_LEVELS]
    x = list(range(len(labels)))

    colors = {"gpt-4o": "#1f77b4", "gpt-4o-mini": "#ff7f0e"}
    for model in MODELS:
        y = [sum(results[model][label]) / len(results[model][label]) for label, _ in DISTRACTOR_LEVELS]
        ax.plot(x, y, marker="o", linewidth=2.5, label=model, color=colors.get(model))
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Distractor Density (# conflicting spec versions injected)", fontsize=11)
    ax.set_ylabel("Avg Answer Quality Score (0–1)", fontsize=11)
    ax.set_title("RAG Agent Degradation Under Adversarial Retrieval\n(multi-hop + conflicting versions, traced with Respan)", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("degradation_curve.png", dpi=150)
    print(f"\n  saved: degradation_curve.png")
    print(f"  traces: platform.respan.ai\n")


if __name__ == "__main__":
    main()
# Rygor Systems — Summary

## What We’re Building

Rygor Systems builds **private, production-grade LLM evaluation packs and release gates** for teams shipping AI systems to real users.

We help teams answer one hard question with confidence:

> **Can we ship this model / prompt / agent change to production — yes or no?**

This is **decision infrastructure**, not annotation, not research, and not public benchmarking.

---

## The Core Problem

LLM teams face three recurring failures:

1. **Public benchmarks don’t predict production**
   - Good benchmark scores ≠ good RAG, tool-use, or safety behavior
   - Public datasets are leaked, overfit, and gamed

2. **Internal evals decay**
   - Built once, never maintained
   - Overfit to prompts
   - No clear ownership
   - Politically hard to enforce

3. **Releases ship without gates**
   - Regressions are caught by users
   - Rollbacks are expensive
   - Confidence is fake

Teams know this is a problem — but nobody wants to own it.

---

## Our Solution

We productize **private evaluations** that act as **release gates**.

### What We Ship
- **Eval Packs**  
  Versioned, private datasets (200–800+ cases) designed around real failure modes  
- **Eval Runner**  
  Lightweight Python harness to run evals and generate a clear scorecard  
- **Release Decision**  
  Binary output: ship / no-ship  
- **Optional Refresh Subscription**  
  Monthly/quarterly new cases to prevent eval decay and overfitting  

This turns evals from “research artifacts” into **production controls**.

---

## Why We Win Now

- Public benchmarks are saturated and unreliable
- Teams ship weekly → eval debt is exploding
- Internal evals are expensive and fragile
- LLM failures are now **business risk**, not just ML issues

Benchmarks = marketing  
Private evals = production safety

---

## Our Wedge Strategy

We start **narrow** with one high-pain area.

**Initial wedge (recommended):**
- **Agent / Tool-Use Evaluations**
  - Function calling correctness
  - Schema adherence
  - Tool selection
  - Retry logic and partial failures

Each wedge becomes a named, versioned SKU:
- `Agent-ToolUse-Eval v0.8`
- `RAG-Grounding-Eval v1.0`
- `Safety-Refusal-Eval 2025Q1`

---

## Why Two Backend Engineers Are Enough

This business is not ML research-heavy.

We do **not** need:
- Model training
- PhDs
- Novel metrics

We **do** need:
- Failure-mode thinking
- Clean schemas
- Deterministic scoring
- CI-style tooling
- Versioning discipline
- Good judgment

Most evals fail because of **bad design**, not bad math.

---

## Target Customers

- Seed–Series C AI startups
- Teams shipping LLMs into production
- Platform / infra / applied AI teams

**Buyers:**
- LLM Engineering Lead
- Head of Platform
- Applied ML Lead
- AI Product Owner

---

## Business Model (Realistic)

- **Eval Pack (one-time)**
  - $3k–$15k (startup scale)
  - $20k–$75k (enterprise scale)

- **Refresh Subscription**
  - $2k–$10k / month

This is **infra spend**, not services or annotation.

---

## North-Star Positioning

> **We build private evaluations that reveal where modern LLMs fail in production — beyond public benchmarks.**

If this sentence stays true, the business works.
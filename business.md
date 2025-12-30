# Rygor Systems — Business & Go-To-Market README

This document covers the **business side** of Rygor Systems:
- Who our customers are
- How we find them
- How we sell
- How customization works (without becoming a services shop)
- What we must learn from clients
- How pricing is structured

This is written to keep us **commercially disciplined**.

---

## 1. Ideal Customers (Who We Build For)

### Primary Customer Profile
- Seed to Series C startups
- Shipping **LLM-powered features to production**
- Fast iteration cycles (weekly / bi-weekly releases)
- Small-to-mid AI teams (3–15 engineers)

### Secondary (Later)
- Infra/platform teams inside larger companies
- AI tooling companies
- Internal AI enablement teams

---

## 2. Buyer Personas (Who Pays)

We do **not** sell to “the company”.
We sell to a **specific pain owner**.

### Primary Buyers
- LLM / AI Engineering Lead
- Applied ML Lead
- Head of Platform / Infra
- AI Product Owner

### What They Care About
- Preventing regressions
- Avoiding public failures
- Faster releases with confidence
- Clear accountability for AI quality

If someone says  
> “We’ll just eyeball outputs”  
they are not a buyer.

---

## 3. Strong Buying Signals

We focus outreach only where pain already exists.

### Strong Signals
- “We ship prompts/models weekly”
- “We’ve had hallucination / tool-use incidents”
- “Our evals are ad-hoc”
- “No one owns eval maintenance”
- “We catch issues after release”

### Weak Signals (Avoid)
- Research-only teams
- Benchmark-focused teams
- “We’ll think about it later”
- “We want a dashboard”

---

## 4. How We Find Clients

### Where We Look
- Job boards (LLM Engineer, AI Platform roles)
- Company blogs shipping agents / RAG / copilots
- GitHub repos with tool-calling / agent code
- Startup demos using LLMs
- Twitter / LinkedIn posts about AI incidents

### Outbound Hook (Simple & Effective)
We do **not** pitch features first.

We ask:
> “What stops a bad prompt or model change from shipping today?”

Follow-ups:
- “How do you catch regressions?”
- “Who owns eval upkeep?”
- “What was your last AI failure?”

If they hesitate → pain exists.

---

## 5. Sales Motion (Low-Friction)

### Step 1: Discovery Call (30 min)
Goal:
- Identify **top 1–2 failure modes**
- Agree on **what ‘bad’ looks like**
- Define **acceptance criteria**

We do NOT:
- Promise platform features
- Agree to unlimited customization

---

### Step 2: Eval Pack Proposal
We propose:
- One named eval SKU
- Fixed scope
- Fixed price
- Clear deliverables

Example:
> “Agent Tool-Use Eval v0.1 — 400 cases, schema adherence + retries”

---

### Step 3: Delivery → Upsell Refresh
Once value is demonstrated:
- Expand case count
- Add categories
- Convert to refresh subscription

---

## 6. Customization Rules (Critical)

Customization is allowed — but **only within product boundaries**.

### Allowed Customization
- New failure categories
- Domain-specific inputs
- Threshold tuning
- Dataset expansion
- Additional wedges (paid)

### Not Allowed (Unless Paid Expansion)
- Unlimited bespoke cases
- Ad-hoc scoring logic
- One-off evals without versioning
- “Just add a few cases” requests

Everything becomes:
- Named
- Versioned
- Priced

---

## 7. What We Learn From Clients

We study **behavior**, not models.

### We Capture
- Allowed vs forbidden actions
- Expected uncertainty handling
- Known failure modes
- Past incidents
- Release cadence
- Risk tolerance

### We Do NOT Promise
- Perfect accuracy
- No false positives
- Model improvement

We promise:
- Better decisions
- Earlier detection
- Clear ownership

---

## 8. Pricing Strategy

We price as **infra**, not services.

### Eval Packs (One-Time)
- Startup pack (200–800 cases): **$3k–$15k**
- Large suite (2k–10k cases): **$20k–$75k**

### Refresh Subscription (Recurring)
- $2k–$10k / month
- Based on:
  - Case volume
  - Update frequency
  - Reporting depth

Discounts only for:
- Design partners
- Case studies
- Early traction

---

## 9. Positioning Discipline

We are **not**:
- Annotation vendors
- Benchmark platforms
- Research tools

We are:
> **Release decision infrastructure for LLM systems**

If a customer wants:
- People → refer out
- Dashboards → say no
- Research → say no

Focus wins.

---

## 10. Internal Rule (For Us)

If something:
- Can’t be versioned
- Can’t be named
- Can’t be priced

👉 We do NOT build it.

This is how we avoid becoming a services shop.
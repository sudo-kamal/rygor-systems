
# Rygor Systems – Technical Deep Dive

Private evaluation infrastructure for:
- LLM **agents** (tool-use, workflows)
- **Indic / Hindi / Hinglish** language behavior

This doc is for two backend SWEs who want a clear architecture, data models, and code skeletons to build an MVP in a few weeks.

---

## 1. High-Level Architecture

### 1.1 Core Idea

You provide **eval packs** (JSON) + a **runner** that customers use to validate their LLM systems before shipping:

- For **agents**: correct tool-use, parameters, and end-state.
- For **Indic**: language understanding, code-mixing, and culturally grounded responses.

### 1.2 Components

```
Eval Pack (JSON)
  - Test cases
  - Tool schemas
  - Expected behaviors

          │
          ▼
Agent / Model Executor
  - Calls LLM or customer agent
  - Captures tool calls & responses
  - Records latency & errors

          │
          ▼
Multi-Layer Evaluator
  - Layer 1: Function existence
  - Layer 2: Parameter presence & types
  - Layer 3: Value ranges & enums
  - Layer 4: Hallucinations (later)
  - Layer 5: State & task completion (later)
  - Indic-specific checks (code-mixing, cultural)

          │
          ▼
Scorecard Generator
  - Per-test results
  - Aggregated pass-rate & failure breakdown
  - SHIP / SHIP_WITH_CAUTION / DO_NOT_SHIP

          │
          ▼
Outputs
  - scorecard.json
  - detailed_failures.json (optional)
```

---

## 2. Data Model

Use Pydantic-style models (even if you don’t actually import pydantic yet).

### 2.1 Enum for failure modes

```python
from enum import Enum

class FailureMode(str, Enum):
    FUNCTION_NOT_EXISTS = "function_not_exists"
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    WRONG_PARAMETER_TYPE = "wrong_parameter_type"
    PARAMETER_VALUE_OUT_OF_RANGE = "parameter_value_out_of_range"
    HALLUCINATED_PARAMETER = "hallucinated_parameter"
    STATE_MISMATCH = "state_mismatch"
    EXECUTION_ERROR = "execution_error"
    INDIC_UNDERSTANDING_FAIL = "indic_understanding_fail"
    CODE_MIX_HANDLING_FAIL = "code_mix_handling_fail"
```

### 2.2 Tool and test schemas

```python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    raw: Optional[Dict[str, Any]] = None

@dataclass
class TestCase:
    id: str
    category: str
    description: str
    user_query: str
    tools_schema: List[ToolSchema]
    expected_failures: Dict[FailureMode, bool]  # which failures are expected in this testcase
    indic_expectations: Optional[Dict[str, Any]] = None  # for Indic tests
    ground_truth_state: Optional[Dict[str, Any]] = None  # for state-based agent tests later
```

### 2.3 Execution and evaluation results

```python
@dataclass
class ExecutionResult:
    test_case_id: str
    success: bool
    tool_calls: List[ToolCall]
    final_response: Optional[str]
    latency_ms: float
    error: Optional[str] = None

@dataclass
class EvalResult:
    test_case_id: str
    passed: bool
    detected_failures: List[FailureMode]
    expected_failures: List[FailureMode]
    severity: str  # "low" | "medium" | "high" | "critical"
    explanation: str
```

---

## 3. Agent / Model Executor

The executor is a thin wrapper that runs **one test case** against a **backend** (either OpenAI, DeepSeek, or a customer’s agent API) and returns an `ExecutionResult`.

### 3.1 OpenAI-style executor (function calling)

```python
# executor.py
import os
import time
import json
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .models import ExecutionResult, ToolCall

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

class AgentExecutor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = httpx.AsyncClient(timeout=30)

    async def execute_llm(self, test_case: Dict[str, Any]) -> ExecutionResult:
        start = time.time()

        tools_schema = test_case["tools_schema"]
        functions = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["parameters"],
            }
            for t in tools_schema
        ]

        try:
            resp = await self.client.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": test_case["user_query"]}],
                    "functions": functions,
                    "function_call": "auto",
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return ExecutionResult(
                test_case_id=test_case["id"],
                success=False,
                tool_calls=[],
                final_response=None,
                latency_ms=(time.time() - start) * 1000,
                error=str(e),
            )

        msg = data["choices"]["message"]
        tool_calls: List[ToolCall] = []

        if "function_call" in msg:
            fc = msg["function_call"]
            args = json.loads(fc.get("arguments", "{}") or "{}")
            tool_calls.append(
                ToolCall(tool_name=fc["name"], parameters=args, raw=fc)
            )

        return ExecutionResult(
            test_case_id=test_case["id"],
            success=True,
            tool_calls=tool_calls,
            final_response=msg.get("content"),
            latency_ms=(time.time() - start) * 1000,
        )
```

### 3.2 Customer agent executor (HTTP endpoint)

Later, you will let customers point you at their own agent:

```python
class HttpAgentExecutor(AgentExecutor):
    def __init__(self, agent_url: str):
        super().__init__(api_key=None)
        self.agent_url = agent_url

    async def execute_agent(self, test_case: Dict[str, Any]) -> ExecutionResult:
        start = time.time()
        try:
            resp = await self.client.post(
                self.agent_url,
                json={
                    "user_query": test_case["user_query"],
                    "tools_schema": test_case["tools_schema"],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return ExecutionResult(
                test_case_id=test_case["id"],
                success=False,
                tool_calls=[],
                final_response=None,
                latency_ms=(time.time() - start) * 1000,
                error=str(e),
            )

        tool_calls = [
            ToolCall(
                tool_name=tc["tool_name"],
                parameters=tc["parameters"],
                raw=tc,
            )
            for tc in data.get("tool_calls", [])
        ]
        return ExecutionResult(
            test_case_id=test_case["id"],
            success=True,
            tool_calls=tool_calls,
            final_response=data.get("final_response"),
            latency_ms=(time.time() - start) * 1000,
        )
```

---

## 4. Multi-Layer Evaluator

The evaluator receives a `TestCase` and an `ExecutionResult` and outputs an `EvalResult`.

### 4.1 Layer overview

- **Layer 1 – Function existence:** Did the agent call a defined tool?
- **Layer 2 – Param presence & types:** Required fields present and correct type?
- **Layer 3 – Param values:** Within ranges, enums, or constraints?
- **Layer 4 – Hallucinations (later):** Params invented vs user input/context?
- **Layer 5 – State (later):** Did the sequence of calls achieve the desired outcome?
- **Indic layers:**  
  - Indic understanding: Did it correctly interpret Hindi/Hinglish semantics?  
  - Code-mix handling: Did it respond in a reasonable mixture/style?

### 4.2 Core evaluator implementation

```python
# evaluator.py
from typing import Dict, Any, List, Set
from .models import TestCase, ExecutionResult, EvalResult, FailureMode

class ToolUseEvaluator:
    def __init__(self):
        self.type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

    def evaluate(self, test_case: Dict[str, Any], exec_result: ExecutionResult) -> EvalResult:
        if not exec_result.success:
            detected = {FailureMode.EXECUTION_ERROR}
            expected = self._expected(test_case)
            return EvalResult(
                test_case_id=test_case["id"],
                passed=detected == expected,
                detected_failures=list(detected),
                expected_failures=list(expected),
                severity="critical",
                explanation=f"Execution error: {exec_result.error}",
            )

        tools_schema = test_case["tools_schema"]
        schema_map = {t["name"]: t for t in tools_schema}

        detected: Set[FailureMode] = set()

        # Layer 1: function existence
        valid_names = set(schema_map.keys())
        for call in exec_result.tool_calls:
            if call.tool_name not in valid_names:
                detected.add(FailureMode.FUNCTION_NOT_EXISTS)

        # Layers 2 & 3
        for call in exec_result.tool_calls:
            if call.tool_name not in schema_map:
                continue
            schema = schema_map[call.tool_name]["parameters"]
            props = schema.get("properties", {})
            required = schema.get("required", [])

            # Required params
            for req in required:
                if req not in call.parameters:
                    detected.add(FailureMode.MISSING_REQUIRED_PARAMETER)

            # Type + range
            for pname, pval in call.parameters.items():
                if pname not in props:
                    continue
                ps = props[pname]
                expected_type = ps.get("type")
                actual_type = self.type_map.get(type(pval))
                if expected_type and actual_type != expected_type:
                    detected.add(FailureMode.WRONG_PARAMETER_TYPE)
                    continue
                if isinstance(pval, (int, float)):
                    if "minimum" in ps and pval < ps["minimum"]:
                        detected.add(FailureMode.PARAMETER_VALUE_OUT_OF_RANGE)
                    if "maximum" in ps and pval > ps["maximum"]:
                        detected.add(FailureMode.PARAMETER_VALUE_OUT_OF_RANGE)

        # Indic check (optional v0): simple heuristic
        if "indic_expectations" in test_case:
            issues = self._check_indic(test_case, exec_result)
            if issues:
                detected.add(FailureMode.INDIC_UNDERSTANDING_FAIL)

        expected = self._expected(test_case)
        severity = self._severity(detected)

        return EvalResult(
            test_case_id=test_case["id"],
            passed=detected == expected,
            detected_failures=list(detected),
            expected_failures=list(expected),
            severity=severity,
            explanation=self._explanation(detected, expected),
        )

    def _expected(self, test_case: Dict[str, Any]) -> Set[FailureMode]:
        return {
            FailureMode(fm)
            for fm, v in test_case.get("expected_failures", {}).items()
            if v
        }

    def _severity(self, detected: Set[FailureMode]) -> str:
        if not detected:
            return "low"
        crit = {
            FailureMode.FUNCTION_NOT_EXISTS,
            FailureMode.EXECUTION_ERROR,
            FailureMode.STATE_MISMATCH,
        }
        if detected & crit:
            return "critical"
        high = {
            FailureMode.MISSING_REQUIRED_PARAMETER,
            FailureMode.WRONG_PARAMETER_TYPE,
            FailureMode.PARAMETER_VALUE_OUT_OF_RANGE,
            FailureMode.HALLUCINATED_PARAMETER,
            FailureMode.INDIC_UNDERSTANDING_FAIL,
        }
        if detected & high:
            return "high"
        return "medium"

    def _explanation(self, detected: Set[FailureMode], expected: Set[FailureMode]) -> str:
        if detected == expected:
            if not detected:
                return "No failures detected; matched expectations."
            return f"Failures {list(detected)} matched expected failures."
        extra = detected - expected
        missing = expected - detected
        parts = []
        if extra:
            parts.append(f"Unexpected failures: {list(extra)}")
        if missing:
            parts.append(f"Expected but not detected: {list(missing)}")
        return " | ".join(parts)

    def _check_indic(self, test_case: Dict[str, Any], exec_result: ExecutionResult) -> List[str]:
        """Very simple placeholder for Indic checks."""
        issues: List[str] = []
        resp = (exec_result.final_response or "").lower()
        indic_exp = test_case.get("indic_expectations") or {}
        if indic_exp.get("should_use_hinglish") and not any(
            w in resp for w in ["hai", "nahi", "bhai", "kyunki"]
        ):
            issues.append("Response not in expected Hinglish style.")
        if indic_exp.get("must_reference_indic_context"):
            hints = indic_exp.get("context_keywords", [])
            if hints and not any(h.lower() in resp for h in hints):
                issues.append("Missing Indic-specific context reference.")
        return issues
```

---

## 5. Scorecard & Runner

### 5.1 Scorecard aggregation

```python
# runner.py
import json
import asyncio
from typing import Dict, Any, List
from .executor import AgentExecutor
from .evaluator import ToolUseEvaluator, FailureMode
from .models import EvalResult

class Scorecard:
    def __init__(self, eval_results: List[EvalResult]):
        self.eval_results = eval_results
        self.total = len(eval_results)
        self.passed = sum(1 for r in eval_results if r.passed)
        self.failed = self.total - self.passed
        self.pass_rate = (self.passed / self.total * 100) if self.total else 0.0
        self.failures_by_type: Dict[str, int] = {}
        for r in eval_results:
            for f in r.detected_failures:
                self.failures_by_type[f.value] = self.failures_by_type.get(f.value, 0) + 1

        # Simple recommendation
        if self.pass_rate >= 95 and self.failures_by_type.get(FailureMode.FUNCTION_NOT_EXISTS.value, 0) == 0:
            self.recommendation = "SHIP"
        elif self.pass_rate >= 85:
            self.recommendation = "SHIP_WITH_CAUTION"
        else:
            self.recommendation = "DO_NOT_SHIP"

    def to_json(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 1),
            "failures_by_type": self.failures_by_type,
            "recommendation": self.recommendation,
        }

class EvalRunner:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.executor = AgentExecutor(api_key=api_key, model=model)
        self.evaluator = ToolUseEvaluator()

    async def run_suite(self, eval_pack: Dict[str, Any]) -> Scorecard:
        results: List[EvalResult] = []
        for tc in eval_pack["test_cases"]:
            exec_result = await self.executor.execute_llm(tc)
            eval_result = self.evaluator.evaluate(tc, exec_result)
            results.append(eval_result)
        return Scorecard(results)
```

### 5.2 CLI

```python
# cli.py
import json
import asyncio
import argparse
from rygor.runner import EvalRunner

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-pack", required=True)
    parser.add_argument("--output", default="scorecard.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    with open(args.eval_pack) as f:
        pack = json.load(f)

    runner = EvalRunner(api_key=None, model=args.model)
    scorecard = await runner.run_suite(pack)
    out = scorecard.to_json()
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    asyncio.run(main())

```

Sources
[1] 10 best LLM evaluation tools with superior integrations in 2025 https://www.braintrust.dev/articles/best-llm-evaluation-tools-integrations-2025
[2] 4. Datadog https://www.confident-ai.com/blog/greatest-llm-evaluation-tools-in-2025
[3] Evaluation and Benchmarking of LLM Agents: A Survey - arXiv https://arxiv.org/html/2507.21504v1
[4] Component-Level Evaluations https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide
[5] Enterprise Guide: Agent Evaluation Frameworks 2025 - Sparkco AI https://sparkco.ai/blog/enterprise-guide-agent-evaluation-frameworks-2025
[6] MILU: A Multi-task Indic Language Understanding Benchmark - ACL ... https://aclanthology.org/2025.naacl-long.507/
[7] LLM Evaluation Guide 2025: Metrics, Framework & Best Practices https://www.xbytesolutions.com/llm-evaluation-metrics-framework-best-practices/
[8] Top 5 LLM Evaluation Tools of 2025 for Reliable AI Systems https://futureagi.com/blogs/top-5-llm-evaluation-tools-2025
[9] INDIC QA BENCHMARK: A Multilingual ... https://aclanthology.org/2025.findings-naacl.141/
[10] How to Evaluate LLMs: Metrics + Best Practices https://galileo.ai/blog/llm-evaluation-step-by-step-guide
[11] The 4 Best LLM Evaluation Platforms in 2025 - LangWatch https://langwatch.ai/blog/the-4-best-llm-evaluation-platforms-in-2025-why-langwatch-eedefines-the-category-with-agent-testing-(with-simulations)
[12] Can your AI reason across the Indic language spectrum? - IBM https://www.ibm.com/think/news/indqa-llm-benchmark
[13] LLM evaluation framework: principles, practices, and tools https://toloka.ai/blog/llm-evaluation-framework-principles-practices-and-tools/
[14] arXiv:2503.16416v1 [cs.AI] 20 Mar 2025 https://arxiv.org/pdf/2503.16416.pdf
[15] Benchmark to evaluate LLMs on low-resource Indic Languages - arXiv https://www.arxiv.org/abs/2512.00333
[16] Building an LLM evaluation framework: best practices - Datadog https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/
[17] Llm Evaluation Tools: Key... https://orq.ai/blog/llm-evaluation-tools
[18] Benchmark to evaluate LLMs on low-resource Indic Languages - Liner https://liner.com/review/indicparam-benchmark-to-evaluate-llms-on-lowresource-indic-languages
[19] LLM Testing in 2025: Top Methods and Strategies https://www.confident-ai.com/blog/llm-testing-in-2024-top-methods-and-strategies
[20] Comparing LLM Evaluation Platforms: Top Frameworks for ... https://arize.com/llm-evaluation-platforms-top-frameworks/

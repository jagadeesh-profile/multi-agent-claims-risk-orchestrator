# Claims-Risk Orchestrator — Evaluation Report

- Generated: 2026-04-30T18:13:40
- Runs per case: 3
- Total LLM-pipeline invocations: 9

## Per-case summary

| Case | Modal action | Agreement | p50 latency | p95 latency | Failures |
|------|--------------|-----------|-------------|-------------|----------|
| A (P_MARY) | `AUTO_APPROVE` | 100% | 44.29s | 44.35s | 0% |
| B (P_ROBERT) | `FLAG_FOR_AUDIT` | 100% | 53.64s | 57.21s | 0% |
| C (P_LINDA) | `ESCALATE_TO_HUMAN` | 100% | 43.73s | 43.83s | 0% |

## Per-agent mean latency (ms)

| Case | ActionAgent | ClaimsAgent | FusionAgent | LabsAgent | NotesAgent | RefinerAgent | ReviewerAgent | RouterAgent |
|---|---|---|---|---|---|---|---|---|
| A | 7417 | 642 | 7970 | 3199 | 3924 | 7023 | 10319 | 2086 |
| B | 6106 | 304 | 21036 | 1536 | 4398 | 7428 | 9827 | 2217 |
| C | 0 | 679 | 10407 | 3189 | 1590 | 8208 | 10799 | 1611 |

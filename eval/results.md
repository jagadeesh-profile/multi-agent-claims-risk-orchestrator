# Claims-Risk Orchestrator — Evaluation Report

- Generated: 2026-04-30T17:30:29
- Runs per case: 3
- Total LLM-pipeline invocations: 9

## Per-case summary

| Case | Modal action | Agreement | p50 latency | p95 latency | Failures |
|------|--------------|-----------|-------------|-------------|----------|
| A (P_MARY) | `AUTO_APPROVE` | 100% | 43.85s | 51.63s | 0% |
| B (P_ROBERT) | `UNKNOWN` | 66% | 82.82s | 263.73s | 0% |
| C (P_LINDA) | `PARSE_ERROR/UNKNOWN` | 33% | 63.24s | 69.34s | 33% |

## Per-agent mean latency (ms)

| Case | ActionAgent | ClaimsAgent | FusionAgent | LabsAgent | NotesAgent | RefinerAgent | ReviewerAgent | RouterAgent |
|---|---|---|---|---|---|---|---|---|
| A | 7651 | 417 | 8021 | 1655 | 6114 | 8991 | 11089 | 2369 |
| B | 7714 | 592 | 108815 | 1529 | 4243 | 8543 | 11191 | 2155 |
| C | 8390 | 1624 | 19753 | 3193 | 1564 | 15970 | 11599 | 1696 |

# LinkedIn Launch Kit — Claims-Risk Orchestrator

This is the content pack to turn the project into ~5 weeks of LinkedIn visibility under the agentic-AI / Google ADK angle. Edit voice freely — the bones are right but they should sound like *you*.

---

## TL;DR strategy

**Positioning:** "The few-people-on-ADK guy who actually shipped a multi-agent pipeline that uses all four primitives." Most agentic-AI content is LangGraph or CrewAI. ADK is under-served. Lean in.

**Cadence:** 1 deep post per week × 5 weeks > 5 shallow posts in one week. The repo is the same; the *angle* changes each post. Don't dump it all at once.

**Format mix:** 4 text posts with one image (architecture diagram or trace screenshot) + 1 short demo video (Loom or native LinkedIn video). Native video gets ~3× the algorithmic reach.

**Tactics:**
- Post Tuesday or Wednesday, 8–10am the timezone where most of your target audience is (US ET if global, IST if India-focused).
- Hashtags: 3 niche, not 10 generic. Use `#ADK` `#AgenticAI` `#VertexAI` `#GoogleCloud` `#MultiAgent` — pick three per post.
- GitHub link goes in the **first comment**, not the post body. LinkedIn down-ranks posts with external links.
- Reply to every comment within the first 90 minutes. Algorithm uses early engagement velocity.
- Tag 2-3 actual ADK practitioners (search "Google ADK" on LinkedIn, find people who post about it). Don't tag random influencers — they unfollow you.

---

## Post 1 — Launch (Week 1)

**Image:** `docs/architecture.svg` exported to PNG at 1080×1080.

**Hook:**

> A Random Forest screams "fraud" on a $24,180 hospital claim.
>
> The discharge note explains why the cost was legitimate.
>
> An averaging system would compromise on a 0.63 score and miss the story entirely.

**Body:**

> I spent the last few weekends building a multi-agent claims-triage pipeline on Google's Agent Development Kit (ADK) — and the case above is the one that justified the whole architecture.
>
> Three model types running in parallel:
>   • Random Forest scoring claim anomaly
>   • TensorFlow DNN scoring lab-panel risk
>   • Gemini extracting signals from free-text discharge notes
>
> Then a Fusion agent that *reasons over disagreement* instead of averaging it. Then a bounded review-and-refine loop that exits early when validation passes. Then a deterministic action layer that writes an audit trail and routes to the right downstream system (auto-payment, care manager, SIU audit, or human review).
>
> What surprised me most: ADK gives you all four agent primitives — Sequential, Parallel, Loop, LlmAgent — and a pipeline like this needs all four doing actual work. I've seen LangGraph builds that simulate the same shape, but ADK's primitives compose cleaner once you accept the framework's opinions.
>
> A couple of things I had to extend:
>   1. A custom `RoutedParallelAgent` so the Router can actually skip specialists, not just fan out to all of them
>   2. A `LoopExitChecker` (BaseAgent subclass, ~30 LOC) that emits `actions.escalate=True` to break out of the review loop the moment we have a passing decision
>
> Synthetic data, runs on a laptop, ~$0.02 per case end-to-end. Code, architecture diagram, and a short demo are below.

**CTA:** "What's your take on Google ADK vs LangGraph for this kind of pipeline?"

**Hashtags:** `#AgenticAI #ADK #VertexAI`

**First comment (GitHub link):**
> Repo with the four-primitive pipeline, the eval harness, and the Loom walkthrough: github.com/<your-handle>/claims-risk-orchestrator

---

## Post 2 — Technique deep-dive: context-aware fusion (Week 2)

**Image:** Screenshot of Case B's audit log entry showing the fusion reasoning.

**Hook:**

> "Just average the model scores" is the most expensive shortcut in multi-agent design.

**Body:**

> Three signals came in for one claim last week (synthetic, but the shape is real):
>   • RF anomaly score: 0.81 — "this claim looks fraudulent"
>   • Labs NN risk score: 0.12 — "patient is fine"
>   • Notes extraction: severity=0.2, billing_consistency=inconsistent
>
> If you average those, you get 0.38. Looks borderline. Auto-route to "monitor."
>
> The actual story: low labs + low severity + *inconsistent billing* means the cost is the anomaly, not the patient. The right answer is FLAG_FOR_AUDIT, not "monitor."
>
> So I gave the Fusion agent explicit reasoning rules instead of a weighted sum:
>   → if claim is high but labs+notes are low and consistent → the claim is the anomaly (boost it)
>   → if labs are missing → drop their weight to zero, cap final confidence at 0.7
>   → if billing_consistency=inconsistent → boost anomaly by 0.15
>   → if all three agree → confidence is high
>
> Took maybe 80 LOC of LlmAgent prompt engineering. The behavioral difference vs averaging is enormous on conflicting cases.
>
> Lesson I keep relearning: when signals disagree, the disagreement *is* the signal.

**Hashtags:** `#AgenticAI #LLM #MachineLearning`

---

## Post 3 — Framework extension: RoutedParallelAgent (Week 3)

**Image:** Code carbon-style screenshot of the ~30-line subclass.

**Hook:**

> Frameworks tell you what NOT to subclass. Sometimes you should anyway.

**Body:**

> Google's ADK ships a `ParallelAgent` that fans out to all child agents concurrently. Beautiful primitive. But it doesn't honor a Router's decision — if the Router says "labs section is missing, skip LabsAgent," ParallelAgent fires it anyway.
>
> So I extended it. ~80 LOC subclass that:
>   1. Reads `routing_decision` from session state (set by the upstream RouterAgent)
>   2. Filters its own `sub_agents` list to only the ones the Router selected
>   3. Falls back to "run them all" if the routing JSON is malformed
>
> The rest of the parallel-fan-out machinery — branch contexts, event merging, py3.11+ async iteration — I borrowed straight from ADK's internals. Three lines of "try the new merge, fall back to the old one" handle the asyncio version skew.
>
> The pattern generalizes: when a framework primitive does 80% of what you need, subclass before you reach for a different framework. The ADK source is small enough to read in one sitting; I'd rather extend it than fork the topology.
>
> Code in the comments.

**Hashtags:** `#GoogleCloud #ADK #SoftwareEngineering`

---

## Post 4 — Small pattern, big payoff: LoopExitChecker (Week 4)

**Image:** Side-by-side trace screenshot — "before: 7 events" vs "after: 3 events" on a Case A run.

**Hook:**

> Your review loop should exit when validation passes. Mine was iterating to max_iter regardless. Here's the 30-LOC fix.

**Body:**

> ADK's `LoopAgent` has a clean exit signal: any sub-agent can yield an event with `actions.escalate=True` and the loop terminates. I wasn't using it.
>
> Original setup:
>   ReviewerAgent → RefinerAgent → ReviewerAgent → RefinerAgent → ReviewerAgent → RefinerAgent
>   (3 iterations, always)
>
> On the happy path — first review passes — that's 4 wasted LLM calls. On Gemini Flash that's pennies, but on a paid model in production it's real money and real latency.
>
> Fix: a `LoopExitChecker(BaseAgent)` that runs *between* Reviewer and Refiner each iteration. It checks `review_result.passed` and `review_result.escalate`, and yields `EventActions(escalate=True)` whenever either is true. The Refiner is skipped, the loop terminates, the next agent in the SequentialAgent picks up.
>
> Total code:
>
>   class LoopExitChecker(BaseAgent):
>     async def _run_async_impl(self, ctx):
>       review = self._parse_review(ctx.session.state.get("review_result"))
>       if review.get("passed") or review.get("escalate"):
>         yield Event(invocation_id=ctx.invocation_id, author=self.name,
>                     actions=EventActions(escalate=True))
>
> Three new ideas I learned from this:
>   1. `EventActions.escalate` isn't an error signal — it's a "we're done here" signal
>   2. A BaseAgent that yields *zero events* is a perfectly valid no-op
>   3. The cleanest pattern for state-machine-style agentic flow is "tiny BaseAgents that gate the loop"
>
> If your agentic pipeline burns LLM calls on no-ops, this is the easiest win.

**Hashtags:** `#AgenticAI #ADK #LLMOps`

---

## Post 4.5 — ML-engineer-flavored variant (alternate Week 4 or Week 6)

> Use this if your target audience leans more "AI/ML engineer" than "agentic developer." Same project, different framing — leads with ML rigor instead of framework primitives.

**Image:** Screenshot of MLflow UI showing the Optuna sweep + the SHAP top-features output from a single ClaimsAgent prediction, side by side.

**Hook:**

> AUC of 0.83 means nothing if your model is poorly calibrated and your fusion layer can't explain why a claim got flagged.

**Body:**

> The agentic side of this project gets the LinkedIn attention, but the ML stack underneath is what makes it actually defensible. Here's what's wired in besides the agents:
>
>   • **Optuna TPE sweep** — 30 trials, 5-fold stratified CV, Bayesian search over `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`. Replaces "I tuned by hand and reported the best run" with a reproducible study.
>
>   • **MLflow tracking** — every training and tuning run logs params, metrics, and artifacts to `./mlruns`. Hiring panels can replay every decision I made.
>
>   • **Calibration metrics** — Brier score (0.0X), log loss, 10-bin reliability curve. AUC is the headline; calibration is whether the headline is honest. A fusion layer that consumes a 0.74 score *as if it were a probability* needs that to actually be a probability.
>
>   • **SHAP per-prediction explanations** — TreeExplainer over the RF, top-3 features attributed for every call from the ClaimsAgent. The LLM gets "score=0.81, top drivers: cost_usd(+0.18), procedure_count(+0.09), drg_code_292(+0.05)" instead of just "score=0.81." That changes what the FusionAgent can reason over.
>
>   • **Model card + data card** — `docs/model_card.md` and `docs/data_card.md` document intended use, evaluation methodology, drift considerations, and limitations. The thing AI/ML hiring panels actually scan for.
>
> The lesson I keep coming back to: agentic patterns are interesting *because* they let you compose interpretable ML with LLM reasoning. Strip out the SHAP and the calibration and the FusionAgent is just an LLM hallucinating about scores it doesn't actually understand.
>
> Code, MLflow runs, and the Optuna study config are linked below.

**Hashtags:** `#MLOps #MachineLearning #ExplainableAI`

---

## Post 5 — Opinion: ADK vs LangGraph after 2 weeks (Week 5)

**Image:** A comparison table — primitives, learning curve, observability, ecosystem.

**Hook:**

> Search "LangGraph" on LinkedIn: 50,000+ posts. Search "Google ADK": maybe 800. After 2 weeks shipping a real pipeline on ADK, I think most of the field is sleeping on it.

**Body:**

> Caveats first: I'm not anti-LangGraph. It's mature, has a huge community, and LangChain's ecosystem around it is unmatched. If you're building today and you need something working tomorrow, pick LangGraph.
>
> But if you're investing in *agentic engineering as a craft*, ADK has three things that surprised me:
>
>   1. **Strict primitive separation.** SequentialAgent, ParallelAgent, LoopAgent, LlmAgent — each does exactly one thing. LangGraph's StateGraph is more powerful but lets you build patterns where you can't tell what's an agent and what's a router. ADK's separation forces clarity.
>
>   2. **Composition is cheap.** A custom subclass is ~30 LOC because the base classes are small. I extended ParallelAgent and BaseAgent in this project — neither was scary.
>
>   3. **Native Gemini integration.** Tool calls, structured output, function-calling — all wired through the `google.genai` types. No translation layer.
>
> What ADK doesn't have yet:
>   • The community. You'll spend more time reading source than reading tutorials.
>   • LangChain's tool ecosystem. You'll write more `FunctionTool` wrappers yourself.
>   • The same observability surface. AgentOps, LangSmith, Phoenix all support LangGraph natively; ADK's still mostly DIY.
>
> If you're a couple of years into agentic systems and you want to stand out: build something real on ADK. The ratio of "people who can talk about it" to "people who've shipped on it" is the most lopsided in the framework space right now.

**Hashtags:** `#AgenticAI #ADK #LangGraph`

---

## Loom demo script (~2:30)

Record at 1080p. Mic the audio. Use the architecture diagram for the intro/outro frames.

```
[0:00–0:15] Frame 1: architecture diagram
"Hi — I'm <name>. I built this multi-agent claims-triage pipeline on
Google ADK. Here's a 2-minute walkthrough of the most interesting case.
Code's linked in the description."

[0:15–0:35] Frame 2: terminal showing Case B input JSON
"This is the input — a $24,180 claim with 6 procedures, but the discharge
note says chest pain was ruled out as non-cardiac. The signals are going
to disagree. Watch what happens."

[0:35–1:15] Frame 3: terminal running `python -m src.main --case B -v`
[Let it run live; narrate while events stream]
"The Router picks all three specialists. They fire in parallel — RF, NN,
notes extraction. Notice the LabsAgent comes back with low risk while
the ClaimsAgent flags high anomaly. Now the Fusion agent reasons over
the disagreement... Reviewer validates... LoopExitChecker emits the
escalate signal early because the first review passed."

[1:15–1:50] Frame 4: terminal showing the final decision JSON
"Final action: FLAG_FOR_AUDIT. Confidence 0.85. Reasoning calls out the
billing inconsistency directly — exactly what we want. Audit log row
written for the SIU queue."

[1:50–2:15] Frame 5: VS Code, agents.py with LoopExitChecker visible
"The architectural primitive that made the early exit work is this
BaseAgent subclass — about 30 lines. Yields actions.escalate=True when
the Reviewer passes, the Refiner gets skipped, the loop terminates."

[2:15–2:30] Frame 6: architecture diagram
"That's the whole pipeline. Sequential, Parallel, Loop, LlmAgent — all
four ADK primitives doing actual work. Code and writeup are linked."
```

---

## Posting checklist (per post)

- [ ] First line is the hook — under 200 characters
- [ ] Image attached (architecture, screenshot, code snippet, or demo frame)
- [ ] Body uses short paragraphs (LinkedIn truncates at ~3 lines on mobile)
- [ ] No external links in the post body — drop them in first comment
- [ ] 3 hashtags max, niche over generic
- [ ] Tag 2 ADK / agentic-AI practitioners (real ones, not influencers)
- [ ] Reply to every comment within first 90 minutes
- [ ] Repost to your story 24 hours later if engagement is good

## What to track

After each post, note the metrics in a spreadsheet — you'll learn which angle resonates fastest:
- Impressions (24h, 7d)
- Reactions, comments, reposts
- Profile views in the 24h after
- Connection requests in the 24h after
- New followers

The post that converts profile visits → follows is your strongest angle. Lean into it for the next 2-3 posts.

---

## What I'd post in the README to support all this

Add a single line near the top:

> "Walkthrough of Case B (the conflict case) on Loom: <link>. Architecture: docs/architecture.svg. LinkedIn launch kit: docs/LAUNCH_KIT.md."

That way anyone who lands on the repo from a LinkedIn click sees the demo immediately. Discovery → engagement → follow loop.

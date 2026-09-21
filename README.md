# Argument Mapper

**Can a fine-tuned 2B open model replace a frontier API for structured argument
extraction, and at what cost?**

Argument extraction is a structured-output task: given an argumentative text,
identify its claims and premises as spans, and the support/attack relations
between them. Frontier APIs do it well. This repository measures what it costs
to stop paying for them — F1, dollars, and latency, with confidence intervals,
on a standard benchmark and an out-of-domain test set.

Every number below is produced by a script in this repository and written under
`results/`. Anything not yet measured says `TBD` rather than an estimate.

---

## Status

| Milestone | State |
|---|---|
| 1. Data and metrics | ✅ complete |
| 2. Frontier baselines (Sonnet 5, Haiku 4.5) | ✅ complete |
| 3. LoRA fine-tune (Qwen3.5-2B) | ✅ complete |
| 4. Constrained decoding (vLLM JSON schema) | ✅ complete |
| 5. Cascade routing | ✅ complete |
| 6. Serving and load test | ⬜ |
| 7. UI | ✅ complete |
| 8. DeBERTa-v3 baseline (stretch) | ⬜ |

---

## Architecture

```mermaid
flowchart LR
    subgraph Data
        AAE[Argument Annotated Essays v2<br/>402 essays]
        MT[arg-microtexts<br/>112 texts, out-of-domain]
    end

    AAE --> CONV[Converters<br/>brat / arggraph]
    MT --> CONV
    CONV --> SCHEMA[(Unified graph schema<br/>components + relations)]

    SCHEMA --> EVAL[Evaluation<br/>component F1 x4, relation F1<br/>document-level bootstrap CIs]

    subgraph Extractors
        CLAUDE[Claude<br/>structured outputs]
        LOCAL[Qwen3.5-2B + LoRA<br/>vLLM, JSON-schema constrained]
        CASC[Cascade<br/>logprob-thresholded routing]
    end

    SCHEMA --> CLAUDE
    SCHEMA --> LOCAL
    LOCAL -. low confidence .-> CLAUDE
    CLAUDE --> CASC
    LOCAL --> CASC

    CLAUDE --> EVAL
    LOCAL --> EVAL
    CASC --> EVAL

    CASC --> API[FastAPI router<br/>batching + content-hash cache]
    API --> UI[React Flow UI]
```

---

## Results

### Datasets

Measured by `argmap-build-datasets`, written to
[`results/datasets/summary.json`](results/datasets/summary.json).

| | AAE v2 | arg-microtexts (en) |
|---|---|---|
| Documents | 402 (258 train / 64 val / 80 test) | 112 (all test) |
| Mean length | 1,974 chars | 423 chars |
| Components | 6,089 | 576 |
| — MajorClaim / Claim / Premise | 751 / 1,506 / 3,832 | not labelled |
| Relations | 3,832 | 464 |
| — supports / attacks | 3,613 / 219 | 290 / 174 |
| Stance (For / Against) | 1,228 / 278 | n/a |

Two properties of this data shape every result that follows:

- **`attacks` is 5.7% of AAE relations** (219 of 3,832), leaving roughly 44 in
  the test set. Per-type attack F1 therefore carries a very wide interval, and
  untyped micro-F1 is the headline number.
- **arg-microtexts has no component type labels.** Its ADUs carry `pro`/`opp`
  dialectical roles, not the MajorClaim/Claim/Premise typology, so typed
  component F1 is reported as N/A there rather than as a fabricated number.

### Alignment ceiling

A model that emits component *text* has not said where that text is. Aligning
it back to a span introduces error that is charged to the extractor, so it caps
achievable F1. Measured by aligning **gold** component text against **gold**
documents, where the right answer is known exactly — 144 AAE documents (val +
test), 2,232 components. Full output in
[`results/alignment/alignment_error.json`](results/alignment/alignment_error.json).

| Condition | Exact span | Exact span (refined) | ≥50% overlap |
|---|---|---|---|
| verbatim | 0.998 | 0.998 | 0.998 |
| whitespace collapsed | 0.998 | 0.998 | 0.998 |
| lowercased | 0.598 | **0.950** | 0.997 |
| reworded (interior tokens dropped) | 0.167 | **0.780** | 0.992 |

Nothing failed to align in any condition (failure rate 0.000 throughout).
arg-microtexts behaves the same or slightly better (reworded: 0.167 → 0.781,
≥50% overlap 1.000).

Three things follow, and they shape how every later result should be read:

1. **Under overlap matching, alignment is not a meaningful ceiling.** Even when
   the text is reworded so that substring search cannot find it, 99.2% of
   components still land within the ≥50% criterion. Overlap-based F1 measures
   extraction, not alignment.
2. **Exact-span F1 largely measures verbatim copying.** It falls from 0.998 to
   0.167 purely from perturbing the text, with no change in extraction quality
   whatsoever. A model that paraphrases correctly is punished as hard as one
   that is wrong. Exact-span numbers are reported for comparability, but
   overlap is the headline.
3. **The refinement pass is worth 4.7×** on paraphrased text (0.167 → 0.780).
   The v1 matcher slides its window in quarter-window steps, so it can only
   land on the true boundary by luck; the hill-climb fixes that.

Two conditions in the JSON — `trimmed` and `truncated_80pct` — are flagged
`exact_span_meaningful: false`. Removing tokens only from the ends leaves a
contiguous substring that `str.find` still locates, so they exercise substring
search rather than alignment, and their true span genuinely differs from the
gold span. They are kept as a degradation check, not read as error.

### Extraction quality

Few-shot (3 exemplars from train), structured outputs, prompt caching.
Intervals are 95% bootstrap over documents. Full reports in
[`results/baselines/`](results/baselines/).

**Argument Annotated Essays, test split (80 documents)**

| Model | Component F1 (overlap, untyped) | Component F1 (overlap, typed) | Relation F1 (overlap) |
|---|---|---|---|
| Claude Haiku 4.5 | 0.859 [0.830, 0.882] | 0.709 [0.673, 0.743] | 0.441 [0.390, 0.494] |
| Claude Sonnet 5 | **0.883** [0.868, 0.897] | **0.762** [0.733, 0.791] | **0.485** [0.428, 0.542] |
| Qwen3.5-2B + LoRA | 0.697 [0.657, 0.735] | 0.518 [0.469, 0.565] | 0.049 [0.028, 0.074] |

**arg-microtexts, out-of-domain (112 documents).** Typed metrics are N/A —
this corpus has no component type labels.

| Model | Component F1 (overlap, untyped) | Relation F1 (overlap) |
|---|---|---|
| Claude Haiku 4.5 | 0.843 [0.810, 0.874] | 0.514 [0.460, 0.567] |
| Claude Sonnet 5 | **0.945** [0.930, 0.960] | **0.676** [0.626, 0.725] |

**Is Sonnet actually better?** Paired bootstrap over the same documents, in
[`results/comparisons/`](results/comparisons/). On AAE test it wins
significantly on 7 of 8 measures — but the headline relation F1 gap is
`+0.044 [-0.001, +0.089]`, which **straddles zero**. Comparing the two
marginal intervals would have missed that; the paired test is what makes the
claim honest.

Three things worth noting:

- **Zero invalid outputs in 548 API calls.** Structured outputs
  (`output_config.format`) made the v1 JSON-repair path entirely unnecessary on
  the Claude route.
- **Only 2 components out of ~7,000 could not be aligned** back to a span, so
  the generate-then-align design costs almost nothing in practice.
- **Relation F1 is the weak half** of both models, at 0.44–0.49 in domain.
  Components are close to solved; relations are not.

### Exact-span F1 is a trap

The alignment ceiling above is not academic. On arg-microtexts, whose spans are
reconstructed from EDUs, exact-span component F1 is **0.025** for Sonnet 5
while overlap F1 on the *same predictions* is **0.945**. The model is finding
the right components and being scored near zero for not reproducing byte
offsets. This is why overlap is the headline metric throughout.

### Cost and latency

Measured per document from the cost ledger
([`results/cost/ledger.jsonl`](results/cost/)), synchronous calls, prompt
caching on. Latency is wall-clock from an ordinary consumer connection.

| Model | $/doc (AAE) | $/1K docs | p50 | p95 |
|---|---|---|---|---|
| Claude Haiku 4.5 | $0.0047 | $4.66 | 3.42 s | 4.68 s |
| Claude Sonnet 5 | $0.0195 | $19.51 | 10.93 s | 24.19 s |

Sonnet 5 costs **4.2×** more and is **3.2×** slower for +0.053 typed component
F1 and +0.044 relation F1. That gap is the whole reason to ask whether a
fine-tuned 2B model can close it.

Prompt caching matters more than it looks: the 3-exemplar prefix is ~4,970
tokens carried on every request, and caching bills it at a tenth. It also has
a trap — see the note on Haiku's 4,096-token minimum in *Reproducing* below.

### Can a fine-tuned 2B replace the API?

Not on this task, and the way it fails is more interesting than the fact that
it does.

The fine-tuned model reaches **79%** of Sonnet 5's untyped component F1 and
**10%** of its relation F1. Identifying which spans are argument components is
evidently learnable from 258 training documents. Wiring those components into
a graph is not.

Looking at its output shows the shape of the failure: it emits a **star**, with
the first component supporting every other one. It has learned what an
argument component looks like and not what an argument *is*.

Two caveats worth stating plainly:

- **This is one recipe, not a verdict on 2B models.** r=16 LoRA, 258 examples,
  10 epochs. A larger adapter, more data, or a relation-specific objective
  might close some of the gap. What is measured here is this configuration.
- **The seeds agree almost exactly** (validation selection score 0.263 ± 0.000
  across three seeds), so the gap is a property of the recipe rather than of
  the random draw.

### Constrained decoding

Same model, same prompts, same greedy decoding. The only difference is whether
vLLM was given the JSON schema. Measured on AAE test;
[`results/constrained/`](results/constrained/).

| | Invalid output | Truncated | Component F1 (overlap, untyped) |
|---|---|---|---|
| Constrained | **0.0%** | 0.0% | 0.697 [0.657, 0.735] |
| Unconstrained | **100.0%** | 28.7% | 0.000 |

Every unconstrained generation failed to parse. **The local route therefore
carries no JSON-repair path** — constrained decoding replaces it outright,
which is the question this milestone asked. The Claude route keeps Pydantic
validation, though it has not needed it either: 548 API calls, zero invalid
outputs.

Two findings about constrained decoding that are easy to miss:

**It guarantees syntax, not termination.** With an unbounded schema the output
was well-formed and still unparseable: the model emitted the *type name* as
each component's id, so every relation referenced the same two generic ids,
and unlimited such relations stay schema-valid. The grammar never required a
closing bracket and generation ran to the token cap. Termination has to be
made a property of the grammar — `maxItems` — and the bound must be reachable
*within* the token budget or it is no bound at all.

**It can impose a convention the model never learned.** Constraining ids to
`^c[0-9]{1,2}$` fixed the numbering completely and lifted relation F1 from
0.003 to 0.028. That it only reached 0.028 is the clearest evidence that the
remaining relation failure is structural rather than a formatting artefact.

### Cascade routing

Run the fine-tuned model on everything; escalate to Claude Haiku 4.5 only where
the local model's mean token logprob falls below a threshold. The threshold is
**swept on validation and reported once on test**. Full sweep in
[`results/cascade/`](results/cascade/).

Cost per document is measured, not estimated: the API price comes from the
spend ledger, and the local price is the A10G hourly rate divided by measured
batched throughput ($0.000205/doc at 1.49 docs/s).

| Escalated | Component F1 | Relation F1 | $/1K docs |
|---|---|---|---|
| 0% (local only) | 0.483 | 0.028 | $0.21 |
| 38% | 0.551 | 0.116 | $1.70 |
| 80% | 0.604 | 0.261 | $3.37 |
| 100% (Claude only) | 0.659 | 0.350 | $4.18 |

**The threshold selected on validation does not transfer to test.** At the
chosen operating point, validation said the cascade matched Claude-only within
its interval. Test says otherwise:

| | Delta vs Claude-only | 95% CI |
|---|---|---|
| Components | −0.056 | [−0.094, −0.025] |
| Relations | −0.093 | [−0.142, −0.046] |

Both intervals exclude zero. The cascade saves **19%** of cost and is
**measurably worse**, so the headline claim a cascade is supposed to
support — same quality, less money — is not available here.

That is the finding, and it is the reason the sweep ran on validation. Had the
threshold been picked on test, the table above would have shown a flattering
operating point that does not exist. With 64 validation documents and a weak
local leg, the threshold fits the validation draw rather than a real property
of the confidence signal.

**Why the saving is small.** A cascade pays in proportion to the traffic its
cheap leg can keep. This one reaches component F1 0.518 against Claude's 0.709,
so it holds about 20% of documents before quality falls away — and 20% of the
traffic is 20% of the bill. A cascade is only as good as its cheap model.

### Throughput

`TBD` — milestone 6.

---

## Interface

![Argument graph extracted from a sample passage](docs/screenshots/02-graph.png)

Source text on the left, argument graph on the right, both linked: selecting a
component highlights it in all three panels. The graph is laid out
bottom-to-top so premises sit beneath what they support and the major claim
rises to the top, which makes an inverted edge obvious at a glance.

Every run shows what it cost — model, latency and dollars — because the
question this repository asks is quality *per dollar*, and a graph with no
price attached hides half of it. Routes the server reports as unconfigured are
disabled rather than hidden, so an unavailable local model is visible instead
of silently missing.

![Component selected, showing its relations in the inspector](docs/screenshots/03-inspector.png)

The gold-annotation overlay draws annotator spans as an underline beneath the
prediction's background tint, rather than as a second background — two
backgrounds are illegible exactly where the layers disagree, which is the part
worth reading closely.

Screenshots are captured by `web/screenshots.mjs` against the production build.
They only ever use the synthetic sample passage: the AAE licence forbids
displaying the corpus, and a screenshot in a public README is about as
displayed as text gets.

---

## Reproducing

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

### Data

`arg-microtexts` downloads automatically. **Argument Annotated Essays v2 cannot
be downloaded headlessly** — TUdatalib sits behind a proof-of-work bot
challenge that returns an HTML page to any non-browser client. Fetch it once in
a browser:

1. Open <https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422>
2. Clear the bot challenge and download `ArgumentAnnotatedEssays-2.0.zip`
3. Place it in `data/raw/`

It is verified by SHA-256 on load. Then:

```bash
uv run argmap-build-datasets            # -> data/processed/, splits/, results/datasets/
uv run python -m argmap.cli.eval_alignment
```

### Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

CI runs all three on CPU against synthetic fixtures, so it never needs the
licensed corpora.

---

## Licensing and data handling

**The corpora are not in this repository and must not be added to it.**

Argument Annotated Essays v2 is distributed under a TU Darmstadt / UKP
agreement restricting it to academic use, whose clause 2.2 forbids publishing,
redistributing, or displaying the data in any form. This repository is public,
so:

- `data/` is gitignored, and a CI job fails the build if corpus files are ever
  tracked
- few-shot exemplars are generated into a gitignored file at runtime rather
  than hardcoded into committed source
- fine-tuned weights are not published, being plausibly derivative
- screenshots use a synthetic passage, never corpus text

If you use the corpora, cite them:

- Christian Stab and Iryna Gurevych. 2016. *Parsing Argumentation Structure in
  Persuasive Essays.* <https://arxiv.org/abs/1604.07370>
- Andreas Peldszus and Manfred Stede. 2015. *An annotated corpus of
  argumentative microtexts.* First European Conference on Argumentation.
  (CC BY-NC-SA 4.0)

The code in this repository is MIT licensed.

---

## Layout

```
src/argmap/
  schema.py        unified graph schema; validates span/text integrity
  align.py         fuzzy text -> span alignment, and its refinement pass
  data/            corpus converters (brat, arggraph), download, splits
  metrics/         matching criteria, P/R/F1, document-level bootstrap
  cli/             dataset build, alignment study
tests/             pytest; synthetic fixtures only, no corpus needed
web/               React Flow + dagre front end
results/           every measured number
splits/            essay ID lists (no text) so splits are reproducible
```

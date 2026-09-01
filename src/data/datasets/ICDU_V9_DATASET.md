# ICDU General Dataset v9 — Clean Core

## Decision summary

V9 replaces v8 as the default SFT dataset on `docker-dev`. The change deliberately
trades synthetic row count for source quality, split integrity, and reproducible
ICDU metadata.

V8 contained 15,191 rows, but it was expanded from only 300 clean prompts. The
expansion introduced near-total validation leakage, repetitive generated suffixes,
forced follow-up questions, unstable labels, and rule-based grammar artifacts. V9
returns to the clean pre-augmentation source rather than attempting to repair those
synthetic variants.

## Files

| File | Records | Use |
|---|---:|---|
| `icdu_training_data_v9.jsonl` | 240 | QLoRA/SFT training |
| `icdu_validation_data_v9.jsonl` | 60 | Checkpoint selection and tuning |
| `icdu_test_data_v9.jsonl` | 54 | Sealed final evaluation only |
| `icdu_v9_source_lineage.jsonl` | 354 | Record-level provenance and hashes |
| `icdu_v9_validation_report.json` | — | Machine-readable integrity results |
| `icdu_v9_manifest.json` | — | Source, split, and file-hash manifest |

`src/config.yaml` points SFT to the v9 train and validation files. The test set is
not part of `DataConfig`; load it separately only for final evaluation.

## Source and split methodology

- The source is the clean Breaking Better v6 messages data already in this
  repository.
- The 300-row v6 training source is deterministically divided into 240 train and
  60 validation records, stratified by persona.
- The independently maintained 54-row v6 validation source becomes the sealed
  v9 test set.
- Split assignment happens at the original prompt level before any future
  augmentation.
- All 354 prompt families are unique and remain in exactly one split.
- The original user prompt and assistant response are preserved verbatim.
- V8 and proactive datasets remain in the repository for history and rollback;
  they are no longer the default SFT input.

## ICDU metadata methodology

Each record retains the existing ten-field ICDU schema:

1. `icdu_id`
2. `persona_archetype`
3. `governing_principle`
4. `capability_layer`
5. `user_intent`
6. `context_summary`
7. `application_prompt`
8. `ideal_response_final`
9. `ideal_response_attributes`
10. `ideal_response_cot`

Metadata is deterministic and conservative:

- Personas use narrow topic and role evidence instead of v8's broad first-match
  rules.
- Intent uses a conservative domain-level objective. The prompt itself carries
  the specific goal, avoiding misleading extraction of phrases such as "I want
  to quit" or "How do I fix it?".
- Governing principle prefers explicit framework headings in the source answer,
  then weighted prompt/answer evidence.
- Capability layer scores Foundational, Transformational, and Aspirational cues;
  it does not fabricate balance.
- `context_summary` embeds the governing principle because the current loader
  consumes context but does not consume `governing_principle` directly.
- `ideal_response_cot` contains a short public rationale outline, not hidden model
  chain-of-thought. The current loader does not consume this field.

## Validation results

- Schema validation: PASS
- Source prompt/response preservation: PASS
- Unique IDs and model units: PASS
- Exact prompt overlap across splits: 0
- Canonical prompt-family overlap across splits: 0
- Known v8 augmentation artifacts: 0
- Total records: 354

## Training implications

This is a clean baseline, not a large production corpus. Because it contains 240
training examples, use conservative SFT settings and monitor for overfitting. Do
not compensate by merging validation or test records into training.

For future expansion, generate only 3–5 reviewed variants per *training* prompt
family, keep every variant in its parent's split, preserve lineage, and leave the
sealed test set unaugmented.

The DPO configuration is intentionally unchanged. DPO uses a separate
messages-format preference source and should be reviewed as its own dataset task.

# Human calibration of the judge committee: what the author must do

**Status: blocked on people.**
- No human label exists yet.
- There is one annotator, the author.
- Double annotation and adjudication need a **second person**. A model must never stand in for one.

## The set

The 76 stratified calibration cases in `evaluation/llm_judge_committee/data/cases.jsonl` (`split == calibration`) were
built by J1's `build_cases.py` (seed 20261123):
- 4 cases per stratum over the 17 strata with at least 20 episodes, plus 8 Hindi or Hinglish cases;
- strata are fault class × read-back × verified completion × completion claimed;
- the cases come from 27 frozen models.

Judges and annotators see the transcript view only. Model identity, fault class and verifier verdict are removed.

## Steps

1. **Author (about 2–3 h).**
   - Open `evaluation/llm_judge_committee/label.html` in a browser and answer the seven questions for all 76 cases.
   - Answers are saved in the browser. Press **Export** and save the file as
     `evaluation/llm_judge_committee/data/human_labels.json`.
   - Do not look at the verifier, the judge outputs or `truth.jsonl` before finishing.
2. **Second annotator (about 2–3 h; any person who has not seen the verifier or the judges).**
   - Open `label_b.html`. It has the same cases in a different order, its own storage, and exports
     `human_labels_b.json`.
   - Give them only the one-paragraph instructions printed on the page.
   - Save the export to `evaluation/llm_judge_committee/data/human_labels_b.json`.
3. **Adjudication (about 30 min, both people together).**
   - Run `python evaluation/llm_judge_committee/build_double_annotation.py`. It writes
     `data/double_annotation_stats.json` (agreement and Cohen's κ per question) and `adjudicate.html`, which lists
     only the disagreements.
   - Decide each disagreement from the transcript, then export to `data/human_labels_adjudicated.json`.
4. **Analysis (seconds).**
   - Run `python evaluation/llm_judge_committee/analyze_j2.py`.
   - Judge metrics against humans use the adjudicated labels when they exist, and otherwise the author's labels,
     marked "single annotator, not adjudicated".
   - Then run J1's `analyze.py` for its own registered human-agreement fields.

## If no second person is available

Report "single annotator; no inter-annotator agreement" (J1 §5 already says so). Do not describe the labels as
adjudicated.

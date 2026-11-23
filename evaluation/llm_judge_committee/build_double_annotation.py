"""Double annotation and adjudication for the 76 stratified calibration cases (PREREGISTRATION_J2.md §4).

Step 1 (always): write `label_b.html` for the SECOND annotator. Same cases, same questions and option order as
`label.html`, but a different case order (seed 20261124), its own browser storage key, and an export named
`human_labels_b.json`. The second annotator must not see the author's labels, the verifier or any judge output.

Step 2 (when both `data/human_labels.json` and `data/human_labels_b.json` exist): write `adjudicate.html`, listing only
the (case, question) pairs on which the two annotators differ, with both answers; the two annotators decide together.
Its export `human_labels_adjudicated.json` = agreed answers + adjudicated answers. Agreement statistics are written to
`data/double_annotation_stats.json`.

No label is ever produced by a model. This script only builds pages and compares files written by people.

    python evaluation/llm_judge_committee/build_double_annotation.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import kappa  # noqa: E402
from build_label_ui import PAGE, QUESTIONS  # noqa: E402

ADJ = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>ActionProof adjudication</title>
<style>body{font:15px/1.5 system-ui,sans-serif;margin:16px;max-width:1200px}pre{white-space:pre-wrap;background:#f4f4f0;
padding:10px;border:1px solid #ccc;font:13px/1.4 ui-monospace,monospace;max-height:420px;overflow:auto}
.item{border-top:2px solid #333;margin-top:18px;padding-top:8px}label{margin-right:12px}</style></head><body>
<h1>Adjudication: <span id="n"></span> disagreements</h1>
<p>Decide each item together, from the transcript only. Then press Export and save the file as
data/human_labels_adjudicated.json.</p><button id="exp">Export</button><div id="list"></div>
<script>
const ITEMS=__ITEMS__, AGREED=__AGREED__, KEY="ap_j2_adjudication";
let D=JSON.parse(localStorage.getItem(KEY)||"{}");
document.getElementById("n").textContent=ITEMS.length;
const L=document.getElementById("list");
for(const it of ITEMS){const d=document.createElement("div");d.className="item";
 d.innerHTML=`<b>${it.id}</b> — ${it.q}<br>author: <b>${it.a}</b>; second annotator: <b>${it.b}</b><br>`+
 it.opts.map(o=>`<label><input type="radio" name="${it.id}|${it.key}" value="${o}" ${D[it.id+"|"+it.key]===o?"checked":""}> ${o}</label>`).join("")+
 `<details><summary>transcript</summary><pre></pre></details>`;d.querySelector("pre").textContent=it.text;L.appendChild(d);}
L.addEventListener("change",e=>{D[e.target.name]=e.target.value;localStorage.setItem(KEY,JSON.stringify(D));});
document.getElementById("exp").onclick=()=>{const out=JSON.parse(JSON.stringify(AGREED));
 for(const [k,v] of Object.entries(D)){const [id,q]=k.split("|");out[id]=out[id]||{};out[id][q]=v;}
 const b=new Blob([JSON.stringify(out,null,1)],{type:"application/json"});const a=document.createElement("a");
 a.href=URL.createObjectURL(b);a.download="human_labels_adjudicated.json";a.click();};
</script></body></html>"""


def main() -> None:
    cases = [json.loads(l) for l in (HERE / "data/cases.jsonl").read_text().splitlines()]
    cal = [{"id": c["case_id"], "lang": c["lang"], "text": c["views"]["transcript"]} for c in cases
           if c["split"] == "calibration"]
    rng = random.Random(20261124)
    order = cal[:]
    rng.shuffle(order)
    page = (PAGE.replace("__CASES__", json.dumps(order, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__QS__", json.dumps(QUESTIONS)).replace('KEY="ap_j1_labels"', 'KEY="ap_j2_labels_b"')
            .replace('a.download="human_labels.json"', 'a.download="human_labels_b.json"')
            .replace("<strong>J1 calibration</strong>", "<strong>J2 second annotator</strong>")
            .replace("ActionProof J1 calibration labels", "ActionProof J2 second-annotator labels"))
    (HERE / "label_b.html").write_text(page)
    print(f"label_b.html: {len(order)} cases (second annotator)")
    pa, pb = HERE / "data/human_labels.json", HERE / "data/human_labels_b.json"
    if not (pa.exists() and pb.exists()):
        print("adjudication: waiting for", [p.name for p in (pa, pb) if not p.exists()])
        return
    A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
    text = {c["id"]: c["text"] for c in cal}
    items, agreed, stats = [], {}, {}
    for key, q, opts in QUESTIONS:
        ids = [i for i in text if key in A.get(i, {}) and key in B.get(i, {})]
        stats[key] = {"n": len(ids), "agreement": sum(A[i][key] == B[i][key] for i in ids) / len(ids) if ids else None,
                      "cohen_kappa": kappa([A[i][key] for i in ids], [B[i][key] for i in ids])}
        for i in ids:
            if A[i][key] == B[i][key]:
                agreed.setdefault(i, {})[key] = A[i][key]
            else:
                items.append({"id": i, "key": key, "q": q, "opts": opts, "a": A[i][key], "b": B[i][key], "text": text[i]})
    (HERE / "data/double_annotation_stats.json").write_text(json.dumps(stats, indent=1))
    (HERE / "adjudicate.html").write_text(ADJ.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
                                          .replace("__AGREED__", json.dumps(agreed)))
    print(f"adjudicate.html: {len(items)} disagreements; stats in data/double_annotation_stats.json")


if __name__ == "__main__":
    main()

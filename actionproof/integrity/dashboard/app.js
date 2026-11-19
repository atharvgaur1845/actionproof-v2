/* ActionProof gateway console. Plain JavaScript, no build step, no libraries.
   Every figure on the page comes from the gateway API; nothing is computed that the API did not return,
   apart from formatting and the grouping of ledger events into rows. */
"use strict";

/* ------------------------------------------------------------------ vocabulary */
const STEP_NAME = { order: "Place order", pay: "Pay supplier", receive: "Receive goods", post: "Post journal entry" };
const STEP_TABLE = { order: "Supplier orders", pay: "Payments", receive: "Goods receipts", post: "Journal entries" };
const TOOL_STEP = { place_supplier_order: "order", pay_supplier: "pay", record_goods_receipt: "receive", post_journal_entry: "post" };
const FAULT_TOOLS = ["pay_supplier", "place_supplier_order", "record_goods_receipt", "post_journal_entry", "refund_payment"];
const MODES = [
  ["passthrough", "Passthrough", "No gateway. Calls go straight to the services and nothing is checked."],
  ["observe", "Observe", "Verifies and records every write. Never changes what happens."],
  ["enforce", "Enforce", "Blocks calls outside the contract, verifies each write, then retries, refunds or escalates."],
  ["enforce_inform", "Enforce and inform", "Enforce, and the agent is shown the verified result instead of the tool's reply."],
];
const MODE_NAME = Object.fromEntries(MODES.map(m => [m[0], m[1]]));
const AGENT_HELP = {
  naive: "Scripted agent that trusts every tool reply and never reads the record back.",
  oracle: "Scripted agent that reads the record after each write and retries or repairs.",
};
const FAULT_KINDS = [
  ["phantom", "Phantom write", "The tool replies success but nothing is stored."],
  ["partial", "Partial write", "The tool replies success but only part of the effect is stored."],
  ["lost_ack", "Lost reply", "The effect is stored but the reply times out."],
  ["precommit_timeout", "Timeout before commit", "The reply times out and nothing is stored."],
  ["dup_delivery", "Duplicate delivery", "The effect is stored twice and the tool replies success."],
  ["never_commit", "Never commits", "Every call to this tool replies success and nothing is ever stored."],
];
const ORD = ["1st", "2nd", "3rd", "4th", "5th"];
const STATE_TEXT = { verified: "verified", absent: "absent", partial: "partial", duplicate: "duplicate", wrong: "wrong",
  in_doubt: "in doubt", escalated: "escalated", pending: "pending" };
const CONTRA_TEXT = { absent: "nothing was stored", partial: "only part of it was stored", duplicate: "it was stored more than once",
  wrong: "a different effect was stored", in_doubt: "the record could not be read fresh" };
const CLAIM_TEXT = { SUPPORTED: "Supported", CONTRADICTED_CLAIM: "Contradicted", UNDERCLAIM: "Under-claim", NO_CLAIM: "No claim" };
const ACTION_TEXT = { SAFE_RETRY: "Safe retry", COMPENSATE: "Compensate", RECONCILE: "Reconcile", ESCALATE: "Escalate to a human",
  NOOP_ALREADY_COMMITTED: "Already verified" };
const POLICY_TEXT = { tool_not_granted: "the contract does not grant this tool",
  entity_out_of_scope: "the call targets a purchase order outside the contract",
  amount_over_limit: "the amount is over the per-operation limit" };

/* ------------------------------------------------------------------ helpers */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}
const $ = (s, el = document) => el.querySelector(s);
const short = x => (x ? String(x).slice(0, 10) : "");
const yn = v => (v ? "yes" : "no");
const human = k => String(k).replace(/_cents$/, " (cents)").replace(/_id$/, "").replace(/_/g, " ");
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem("ap." + k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem("ap." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
function money(cents, currency) {
  if (cents === null || cents === undefined) return null;
  try {
    return new Intl.NumberFormat("en-US", { style: "currency", currency: currency || "USD" }).format(cents / 100);
  } catch { return (cents / 100).toFixed(2); }
}
const int = n => (typeof n === "number" ? n.toLocaleString("en-US") : String(n));
function fmtNum(v, dp) {
  if (typeof v !== "number") return v === null || v === undefined ? "none" : String(v);
  if (dp !== undefined) return v.toFixed(dp);
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  return dp === undefined ? String(+v.toFixed(4)) : v.toFixed(dp);
}
function clock(t) {
  const d = new Date(t * 1000);
  return d.toLocaleTimeString("en-GB", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

class ApiError extends Error {
  constructor(endpoint, status, detail) {
    super(status ? `${endpoint} returned ${status}: ${detail}` : `${endpoint} failed: ${detail}`);
    this.endpoint = endpoint; this.status = status; this.detail = detail;
  }
}
async function api(method, path, body) {
  const endpoint = `${method} ${path}`;
  let r;
  try {
    r = await fetch(path, { method, headers: body ? { "content-type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined });
  } catch (e) { throw new ApiError(endpoint, 0, "the gateway did not respond (" + e.message + ")"); }
  let data = null;
  try { data = await r.json(); } catch { /* non JSON body */ }
  if (!r.ok) {
    let d = data && data.detail !== undefined ? data.detail : r.statusText;
    if (Array.isArray(d)) d = d.map(x => `${(x.loc || []).join(".")}: ${x.msg}`).join("; ");
    throw new ApiError(endpoint, r.status, d);
  }
  return data;
}
function errorBox(e, what) {
  return h("div", { class: "error", role: "alert" },
    h("b", {}, what || "Request failed"), h("div", { class: "mono" }, e.message || String(e)));
}
function skeleton(lines) {
  return h("div", { class: "panel sk-block", "aria-busy": "true", "aria-label": "Loading" },
    h("div", { class: "sk h" }), ...(lines || ["w80", "w60", "w80", "w30", "w60"]).map(c => h("div", { class: "sk " + c })));
}

/* ------------------------------------------------------------------ state */
const S = {
  view: "transaction", scenarios: null, scenErr: null,
  tx: null, events: [], txErr: null, txLoading: false, selSeq: null, confirmTamper: false,
  verify: null, verifyErr: null,
  compare: null, compareErr: null, compareLoading: false,
  stream: { es: null, status: "connecting", last: 0, events: [], filter: "", reconnects: 0, seen: new Set() },
  evaluation: null, evalErr: null,
};

/* ------------------------------------------------------------------ form */
function scenario() {
  const v = $("#f-scenario").value;
  return (S.scenarios || []).find(s => `${s.split}/${s.task_id}` === v);
}
function formBody() {
  const sc = scenario();
  const f = new FormData($("#form"));
  const b = { task_id: sc.task_id, split: sc.split, agent: f.get("agent"), mode: f.get("mode") };
  const src = f.get("faultsrc");
  if (src === "none") { b.faults = []; b.never_commit = null; }
  else if (src === "custom") {
    const kind = $("#f-kind").value, tool = $("#f-tool").value;
    if (kind === "never_commit") { b.faults = []; b.never_commit = tool; }
    else { b.faults = [{ tool, k: +$("#f-k").value, kind }]; b.never_commit = null; }
  }
  return b;
}
function saveForm() {
  const f = new FormData($("#form"));
  store.set("form", { scenario: $("#f-scenario").value, agent: f.get("agent"), mode: f.get("mode"), faultsrc: f.get("faultsrc"),
    tool: $("#f-tool").value, k: $("#f-k").value, kind: $("#f-kind").value });
}
function buildForm() {
  const saved = store.get("form") || {};
  $("#f-modes").append(...MODES.map(([v, name, help]) => h("label", { class: "opt" },
    h("input", { type: "radio", name: "mode", value: v, checked: v === (saved.mode || "enforce") }),
    h("b", {}, name), h("span", {}, help))));
  if (saved.agent) { const r = $(`input[name=agent][value="${saved.agent}"]`); if (r) r.checked = true; }
  $("#f-tool").append(...FAULT_TOOLS.map(t => h("option", { value: t }, t)));
  $("#f-k").append(...ORD.slice(0, 4).map((o, i) => h("option", { value: i }, `${o} call`)));
  $("#f-kind").append(...FAULT_KINDS.map(([v, n]) => h("option", { value: v }, n)));
  if (saved.tool) $("#f-tool").value = saved.tool;
  if (saved.k) $("#f-k").value = saved.k;
  if (saved.kind) $("#f-kind").value = saved.kind;
  renderFaultSrc(saved.faultsrc || "scenario");
  $("#form").addEventListener("change", e => {
    if (e.target.name === "faultsrc" || e.target.id === "f-scenario") renderFaultSrc(new FormData($("#form")).get("faultsrc"));
    syncForm(); saveForm();
  });
  $("#form").addEventListener("submit", e => {
    e.preventDefault();
    if (!scenario()) return;
    if (S.view === "compare") runCompare(); else runTx();
  });
  syncForm();
}
function renderFaultSrc(sel) {
  const sc = scenario();
  const own = sc ? (sc.fault_class === "clean" ? "none (clean scenario)" : sc.fault_class) : "loading";
  const box = $("#f-faultsrc");
  box.replaceChildren(...[
    ["scenario", "Scenario's own fault", `This scenario ships with: ${own}.`],
    ["none", "No fault", "Every service behaves correctly."],
    ["custom", "Custom fault", "Choose the tool, which call and what goes wrong."],
  ].map(([v, n, help]) => h("label", { class: "opt" },
    h("input", { type: "radio", name: "faultsrc", value: v, checked: v === sel }), h("b", {}, n), h("span", {}, help))));
  $("#f-custom").hidden = sel !== "custom";
}
function syncForm() {
  const sc = scenario();
  $("#f-prompt").textContent = sc ? sc.prompt : "";
  $("#f-agent-help").textContent = AGENT_HELP[new FormData($("#form")).get("agent")] || "";
  const kind = $("#f-kind").value;
  $("#f-k").disabled = kind === "never_commit";
  $("#f-kind-help").textContent = (FAULT_KINDS.find(k => k[0] === kind) || [])[2] || "";
  const cmp = S.view === "compare";
  const btn = $("#f-submit");
  btn.textContent = cmp ? "Compare all four modes" : "Run transaction";
  btn.disabled = !sc || S.txLoading || S.compareLoading;
  $("#f-submit-help").textContent = cmp
    ? "Runs the same scenario, agent and fault once under each mode. Each run is a real execution."
    : "Runs the scripted agent once through the gateway in the selected mode.";
}
async function loadScenarios() {
  const sel = $("#f-scenario");
  try {
    S.scenarios = await api("GET", "/v1/scenarios");
    S.scenErr = null;
  } catch (e) {
    S.scenErr = e;
    sel.replaceChildren(h("option", {}, "Scenarios unavailable"));
    $("#f-prompt").replaceChildren(errorBox(e, "Could not load scenarios"));
    return;
  }
  if (!S.scenarios.length) { sel.replaceChildren(h("option", {}, "No scenarios")); return; }
  const groups = {};
  for (const s of S.scenarios) (groups[s.split] = groups[s.split] || []).push(s);
  const label = { proc_dev: "Development", proc_val: "Validation", proc_heldout_service: "Held-out services" };
  sel.replaceChildren(...Object.entries(groups).map(([split, list]) => h("optgroup", { label: `${label[split] || split} (${split})` },
    list.map(s => h("option", { value: `${s.split}/${s.task_id}` },
      `${s.task_id}  ${s.requested.join(", ")}  (${s.fault_class})`)))));
  const saved = (store.get("form") || {}).scenario;
  const def = S.scenarios.find(s => `${s.split}/${s.task_id}` === saved) || S.scenarios.find(s => s.fault_class === "phantom") || S.scenarios[0];
  sel.value = `${def.split}/${def.task_id}`;
  sel.disabled = false;
  renderFaultSrc(new FormData($("#form")).get("faultsrc") || "scenario");
  syncForm();
  if (S.view === "transaction" && !S.tx) renderTx();
}

/* ------------------------------------------------------------------ ledger event text */
function replyText(obs) {
  if (!obs || typeof obs !== "object") return String(obs);
  const parts = [obs.status || "no status"];
  if (obs.error) parts.push(obs.error);
  for (const [k, v] of Object.entries(obs)) {
    if (["status", "error", "detail", "observed"].includes(k)) continue;
    if (Array.isArray(v)) parts.push(`${v.length} ${human(k)}`);
    else if (k === "as_of") parts.push(`version ${v}`);
    else if (v === true) parts.push(human(k));
    else if (v !== null && typeof v !== "object") parts.push(`${human(k)} ${v}`);
  }
  return parts.join(" · ");
}
function policyText(r) {
  if (!r) return "";
  if (r.startsWith("commit_point_unmet:")) {
    return "an earlier step it depends on is not verified yet (" + r.split(":")[1].split(",").map(s => STEP_NAME[s] || s).join(", ") + ")";
  }
  return POLICY_TEXT[r] || r;
}
function attemptOf(op) { const m = /:(\d+)$/.exec(op || ""); return m ? +m[1] + 1 : null; }
function describe(e, currency) {
  const b = e.body || {};
  if (b.tampered) return { what: "Edited after the fact", det: b.note || "stored content changed", cls: "tampered" };
  const step = STEP_NAME[b.step] || b.step;
  switch (e.kind) {
    case "contract_bound":
      return { what: "Contract bound", det: `${b.contract_id}, ${MODE_NAME[b.mode] || b.mode} mode, hash ${short(b.content_hash)}` };
    case "call_intercepted":
      return { what: `Intercepted ${b.tool}`, det: Object.entries(b.args || {}).map(([k, v]) => `${k} ${v}`).join(", ") };
    case "policy_decision":
      return { what: `Blocked ${b.tool}`, det: `Refused because ${policyText(b.reason)}.`, tag: ["rejected", "esc"] };
    case "dispatched":
      return { what: `Dispatched ${b.tool}`, det: `Attempt ${attemptOf(e.op_id) || 1}. Service version ${b.dispatch_version} at dispatch.` };
    case "ack_received":
      return { what: "Reply logged as a claim", det: `${replyText(b.ack)}. Not treated as proof.` };
    case "evidence":
      return { what: `Read ${(STEP_TABLE[b.step] || b.step || "").toLowerCase()}`,
        det: `${(b.observed || []).length} ${(b.observed || []).length === 1 ? "row" : "rows"} at version ${b.as_of_version}${b.fresh === false ? ", stale" : ""}` };
    case "verdict": {
      const ok = b.state === "verified";
      return { what: `Verdict: ${step}`, tag: [STATE_TEXT[b.state] || b.state, ok ? "ok" : b.state === "escalated" ? "esc" : "bad"] };
    }
    case "recovery_decision": {
      const a = b.action;
      let det = "";
      if (a === "SAFE_RETRY") det = `Verified absent after dispatch, so a retry cannot duplicate it. Key ${b.reuse_key}.`;
      else if (a === "COMPENSATE") det = `Refund ${b.refund_of} for ${money(b.amount_cents, currency)}.`;
      else if (a === "RECONCILE") det = `Record the missing quantity of ${b.record_missing_qty}.`;
      else if (a === "ESCALATE") det = `${b.reason}. Unrecovered money ${money(b.residual_cents || 0, currency)}.`;
      else if (a === "NOOP_ALREADY_COMMITTED") det = "Step already verified. Replayed the stored result, no new write.";
      return { what: `${ACTION_TEXT[a] || a}${b.step ? ": " + step : ""}`, det, tag: a === "ESCALATE" ? ["escalated", "esc"] : null };
    }
    case "claim":
      return { what: "Agent's report recorded", det: `Status ${(b.report || {}).status}. Checked against the record, not trusted.` };
    case "finalized":
      return { what: `Finalized: ${b.status}`,
        det: `Claim ${b.claim}. Exposure ${money(b.exposure_cents, currency)}. Ledger head ${short(b.ledger_head)}.`,
        tag: [b.status, b.status === "completed" ? "ok" : b.status === "escalated" ? "esc" : "bad"] };
    default:
      return { what: e.kind, det: JSON.stringify(b).slice(0, 160) };
  }
}
function evNode(e, currency) {
  const d = describe(e, currency);
  return h("div", { class: "ev" + (d.cls ? " " + d.cls : ""), "data-seq": e.seq },
    h("span", { class: "seq" }, "#" + e.seq), h("span", { class: "what" }, d.what),
    d.tag ? h("span", { class: "tag " + d.tag[1] }, d.tag[0]) : null,
    d.det ? h("span", { class: "det" }, d.det) : null);
}

/* ------------------------------------------------------------------ transaction view */
function segmentsOf(d, events) {
  const segs = [{ kind: "setup", items: [] }];
  let cur = segs[0], ai = 0;
  for (const e of events) {
    if (e.kind === "call_intercepted") {
      cur = { kind: "call", n: ai + 1, call: d.agent_view[ai], head: e, items: [] };
      ai++; segs.push(cur);
    } else cur.items.push(e);
  }
  if (d.mode === "passthrough") d.agent_view.forEach((a, i) => segs.push({ kind: "call", n: i + 1, call: a, items: [] }));
  for (const s of segs) {
    const rows = [];
    let row = null;
    for (const e of s.items) {
      if (e.kind === "evidence") { row = { gw: [], sor: e }; rows.push(row); continue; }
      if (e.kind === "verdict" && row && row.sor && !row.gw.length) { row.gw.push(e); continue; }
      if (!row || row.sor) { row = { gw: [], sor: null }; rows.push(row); }
      row.gw.push(e);
    }
    s.rows = rows.length ? rows : [{ gw: [], sor: null }];
  }
  return segs;
}
function sorNode(e) {
  const b = e.body || {};
  if (b.tampered) return evNode(e);
  const rows = b.observed || [];
  const head = h("div", { class: "sor-h" }, h("span", { class: "seq" }, "#" + e.seq),
    `${STEP_TABLE[b.step] || b.step} `, h("span", { class: "mono" }, `version ${b.as_of_version}${b.fresh === false ? ", stale read" : ""}`));
  if (!rows.length) return h("div", {}, head, h("div", { class: "norows" }, "No rows. Nothing is stored."));
  const cols = [...new Set(rows.flatMap(r => Object.keys(r)))].filter(c => c !== "po_id");
  return h("div", {}, head, h("table", { class: "rows" },
    h("thead", {}, h("tr", {}, cols.map(c => h("th", {}, human(c))))),
    h("tbody", {}, rows.map(r => h("tr", {}, cols.map(c => h("td", {}, r[c] === null || r[c] === undefined ? "none" : String(r[c]))))))));
}
function agentNode(seg, d) {
  if (seg.kind === "setup") {
    return h("div", {}, h("div", { class: "note" }, "Task given to the agent"), h("div", { class: "quote clamp", title: d.prompt }, d.prompt));
  }
  const a = seg.call;
  if (!a) return h("div", { class: "na" }, "No matching agent call");
  const step = TOOL_STEP[a.tool];
  const firstVerdict = seg.items.find(e => e.kind === "verdict");
  const vIdx = firstVerdict ? seg.items.indexOf(firstVerdict) : -1;
  const evidence = vIdx > 0 && seg.items[vIdx - 1].kind === "evidence" ? seg.items[vIdx - 1] : null;
  const state = firstVerdict && firstVerdict.body ? firstVerdict.body.state : null;
  const out = [h("div", { class: "tool" }, a.tool),
    h("div", { class: "args" }, Object.entries(a.args || {}).map(([k, v]) => `${k} ${v}`).join(", "))];

  if (a.tool === "submit_report") {
    const rep = a.args || {};
    let struck = false, note = null;
    if (d.mode === "passthrough") {
      struck = rep.status === "completed" && d.truth.agent_level.false_success;
      if (struck) note = h("div", { class: "note pen" }, "Contradicted by the truth verifier: the requested work does not hold.");
    } else {
      const c = d.disposition.claim;
      struck = c === "CONTRADICTED_CLAIM";
      if (struck) note = h("div", { class: "note pen" }, "Contradicted by the gateway's final check of the record.");
      else if (c === "UNDERCLAIM") note = h("div", { class: "note" }, `Under-claim: the work holds but the agent reported ${rep.status}.`);
      else if (c === "SUPPORTED") note = h("div", { class: "note ink" }, "Supported by the gateway's final check of the record.");
    }
    out.push(h("div", { class: "reply" + (struck ? " struck" : "") }, h("span", { class: "k" }, "Report"), h("span", { class: "val" }, rep.status)));
    if (rep.summary) out.push(h("div", { class: "quote" }, rep.summary));
    if (note) out.push(note);
    return h("div", {}, out);
  }

  let contradicted = false, why = null;
  const ok = a.obs && a.obs.status === "ok";
  if (ok && !a.refused && !a.replayed) {
    if (state && state !== "verified") {
      contradicted = true;
      why = `Contradicted by the system of record${evidence ? ` (event #${evidence.seq})` : ""}: ${CONTRA_TEXT[state] || state}.`;
    } else if (d.mode === "passthrough" && step && d.contract.requested.includes(step)) {
      const t = d.truth.system_level.steps[step];
      if (t && t !== "VERIFIED_SUCCESS") {
        contradicted = true;
        why = `Contradicted by the final system of record: the truth verifier finds this step ${t.toLowerCase().replace(/_/g, " ")}.`;
      }
    }
  }
  out.push(h("div", { class: "reply" + (contradicted ? " struck" : "") }, h("span", { class: "k" }, "Reply"), h("span", { class: "val" }, replyText(a.obs))));
  if (contradicted) out.push(h("div", { class: "note pen" }, why));
  if (a.refused) out.push(h("div", { class: "note" }, `Refused by the gateway: ${policyText(a.refused)}.`));
  if (a.replayed) out.push(h("div", { class: "note" }, "Answered from the already verified result. No new write was made."));
  if (a.obs && a.obs.error === "timeout" && state === "verified") {
    out.push(h("div", { class: "note ink" }, `The reply timed out, yet the record shows the write stored${evidence ? ` (event #${evidence.seq})` : ""}.`));
  }
  if (contradicted && d.mode !== "passthrough") {
    const later = seg.items.filter(e => e.kind === "verdict").slice(1).find(e => e.body && e.body.state === "verified");
    const acts = seg.items.filter(e => e.kind === "recovery_decision").map(e => ACTION_TEXT[e.body.action] || e.body.action);
    if (later) out.push(h("div", { class: "note ink" }, `The gateway recovered (${acts.join(", ").toLowerCase()}) and verified the step at event #${later.seq}.`));
    else if (acts.length) out.push(h("div", { class: "note" }, `The gateway responded with: ${acts.join(", ").toLowerCase()}.`));
  }
  if (d.mode === "enforce_inform" && firstVerdict && !a.refused) {
    out.push(h("div", { class: "note" }, "In this mode the agent was shown the gateway's verdict instead of this reply."));
  }
  return h("div", {}, out);
}
function lanesNode(d, events) {
  const cur = d.contract.limits.currency;
  const g = h("div", { class: "lanes", role: "table", "aria-label": "Agent, gateway and system of record, aligned over time" });
  const place = (el, row, col, span) => { el.style.gridRow = span > 1 ? `${row} / span ${span}` : String(row); el.style.gridColumn = String(col); return el; };
  g.append(
    place(h("div", { class: "lh gut", role: "columnheader" }, h("div", { class: "n" }, "Time")), 1, 1),
    place(h("div", { class: "lh", role: "columnheader" }, h("div", { class: "n" }, "Agent said"),
      h("div", { class: "d" }, "Each call the agent made and the reply the tool returned")), 1, 2),
    place(h("div", { class: "lh gw", role: "columnheader" }, h("div", { class: "n" }, "Gateway did"),
      h("div", { class: "d" }, d.mode === "passthrough" ? "Passthrough: nothing is intercepted or verified" : "Policy, dispatch, verdicts, recovery and finalization")), 1, 3),
    place(h("div", { class: "lh sor", role: "columnheader" }, h("div", { class: "n" }, "System of record shows"),
      h("div", { class: "d" }, "Authoritative rows the verifier read from each service")), 1, 4));
  let r = 2;
  const segs = segmentsOf(d, events);
  segs.forEach((s, si) => {
    const span = s.rows.length;
    const seqs = [s.head, ...s.items].filter(Boolean).map(e => e.seq);
    const isReport = s.call && s.call.tool === "submit_report";
    const label = s.kind === "setup" ? "Setup" : isReport ? "Report" : `Call ${s.n}`;
    const range = seqs.length ? (seqs.length > 1 ? `#${seqs[0]}–${seqs[seqs.length - 1]}` : `#${seqs[0]}`) : "";
    const first = si === 0 ? " first-row" : "";
    g.append(place(h("div", { class: "cell gut seg-start" + first }, h("b", {}, label), range), r, 1, span));
    g.append(place(h("div", { class: "cell seg-start" + first }, agentNode(s, d)), r, 2, span));
    s.rows.forEach((row, i) => {
      const st = i === 0 ? " seg-start" + first : "";
      let gw;
      if (d.mode === "passthrough" && s.kind === "call") gw = h("div", { class: "cell na" + st }, i === 0 && si === 1 ? "No gateway in this mode." : "");
      else gw = h("div", { class: "cell" + st }, row.gw.map(e => evNode(e, cur)));
      g.append(place(gw, r + i, 3));
      g.append(place(h("div", { class: "cell" + st }, row.sor ? sorNode(row.sor) : null), r + i, 4));
    });
    r += span;
  });
  // the independent truth verifier's reading of the final state
  const sys = d.truth.system_level;
  g.append(place(h("div", { class: "cell gut seg-start" }, h("b", {}, "Final state")), r, 1));
  g.append(place(h("div", { class: "cell seg-start na" }, d.agent_report ? `Agent reported: ${d.agent_report.status}` : "The agent did not report."), r, 2));
  g.append(place(h("div", { class: "cell seg-start na" }, d.mode === "passthrough" ? "" : `Gateway disposition: ${d.disposition.status}`), r, 3));
  g.append(place(h("div", { class: "cell seg-start" },
    h("div", { class: "sor-h" }, "Truth verifier, independent of the gateway, reads the final state"),
    h("div", { class: "steps", style: "margin:.43rem 0 0" }, Object.entries(sys.steps).map(([k, v]) =>
      h("span", { class: "tag " + (v === "VERIFIED_SUCCESS" ? "ok" : "bad") }, `${STEP_NAME[k] || k}: ${v}`)))), r, 4));
  return g;
}
function contractNode(d) {
  const c = d.contract, e = c.expected, l = c.limits;
  const item = (k, v, sub, cls) => h("div", { class: cls || "" }, h("dt", {}, k), h("dd", {}, v, sub ? h("small", {}, sub) : null));
  return h("dl", { class: "panel contract", "aria-label": "Bound contract" },
    item("Requested steps", c.requested.map(s => STEP_NAME[s] || s).join(", "), null, "wide sans"),
    item("Purchase order", e.po_id),
    item("Item code", e.sku || "none"),
    item("Expected quantity", int(e.qty)),
    item("Expected amount", money(e.amount_cents, l.currency), `${int(e.amount_cents)} cents`),
    item("Payee account", e.payee_account || "none"),
    item("Limit per operation", money(l.per_op_max_cents, l.currency), `total ${money(l.total_max_cents, l.currency)}`),
    item("Write attempts", int(c.max_write_attempts), c.allow_compensation ? "refunds allowed" : "refunds not allowed"),
    h("div", { class: "hash" }, h("dt", {}, "Contract hash (SHA-256)"), h("dd", { title: d.contract_hash }, d.contract_hash.slice(0, 16) + "…", h("small", {}, c.contract_id))));
}
function outcomeNode(d) {
  const cur = d.contract.limits.currency, sys = d.truth.system_level, pt = d.mode === "passthrough";
  const disp = d.disposition || {};
  const dispCls = disp.status === "completed" ? (sys.vsc ? "ink" : "pen") : "";
  const claim = disp.claim;
  const box = (l, v, cls, s) => h("div", {}, h("div", { class: "l" }, l), h("div", { class: "v " + (cls || "") }, v), h("div", { class: "s" }, s));
  return h("div", { class: "panel outcome" },
    box("Final disposition", disp.status || "none", dispCls,
      pt ? "The agent's own report. No gateway checked it." : "The gateway's decision after its final check of the record."),
    pt ? box("Claim check", "Not checked", "", `No gateway. The truth verifier finds the agent's claim ${d.truth.agent_level.false_success ? "false" : "not false"}.`)
      : box("Claim check", CLAIM_TEXT[claim] || claim, claim === "SUPPORTED" ? "ink" : claim === "CONTRADICTED_CLAIM" ? "pen" : "",
        h("span", {}, "Agent's report against the record. ", h("span", { class: "mono" }, claim))),
    pt || disp.exposure_cents === undefined
      ? box("Money exposure", "Not measured", "", "Exposure is measured by the gateway, which is absent in passthrough.")
      : box("Money exposure", money(disp.exposure_cents, cur), disp.exposure_cents > 0 ? "pen" : "", "Paid out without a verified payment step."),
    h("div", {}, h("div", { class: "l" }, "Truth verifier"),
      h("div", { class: "v " + (sys.vsc ? "ink" : "pen") }, sys.vsc ? "Verified" : "Not verified"),
      h("div", { class: "s" }, "Verified completion ", h("b", { class: "mono" }, yn(sys.vsc)), ", false finalization ",
        h("b", { class: "mono" + (sys.false_success ? " pen" : "") }, yn(sys.false_success)), ", duplicate payment ",
        h("b", { class: "mono" + (sys.duplicate_irreversible ? " pen" : "") }, yn(sys.duplicate_irreversible)), ".")));
}
function stepsNode(d) {
  const sys = d.truth.system_level.steps, gws = (d.disposition || {}).steps || {};
  return h("div", { class: "steps", "aria-label": "Step status" }, d.contract.requested.map(s => {
    const t = sys[s], g = gws[s];
    const cls = t === "VERIFIED_SUCCESS" ? "ok" : g === "escalated" ? "esc" : "bad";
    return h("span", { class: "chip " + cls }, h("b", {}, STEP_NAME[s] || s),
      h("span", { class: "mono" }, `${g ? "gateway " + (STATE_TEXT[g] || g) + ", " : ""}truth ${t ? t.toLowerCase().replace(/_/g, " ") : "unknown"}`));
  }));
}
function chainNode(d) {
  const ev = S.events, v = S.verify;
  const bad = v && !v.ok ? v.first_bad_seq : null;
  const ctl = h("div", { class: "chain-ctl" });
  const vbtn = h("button", { class: "btn sm", type: "button", onclick: verifyChain }, "Verify chain");
  let res;
  if (S.verifyErr) res = h("span", { class: "res pen" }, S.verifyErr.message);
  else if (!v) res = h("span", { class: "res muted" }, "Recomputes every hash in the ledger from the first event.");
  else if (v.ok) res = h("span", { class: "res" }, h("b", { class: "ink" }, "Intact. "), `${int(v.n)} events recomputed. Head `, h("span", { class: "mono" }, short(v.head)), ".");
  else {
    const inTx = ev.some(e => e.seq === bad);
    res = h("span", { class: "res" }, h("b", { class: "pen" }, `Broken at event #${bad}. `),
      `Its content no longer matches its hash${v.anchor_mismatch ? " or an anchor" : ""}. ${inTx ? "Highlighted below." : "That event belongs to another transaction."}`);
  }
  const opts = ev.map(e => h("option", { value: e.seq, selected: e.seq === S.selSeq }, `#${e.seq} ${e.kind}`));
  const sel = h("select", { "aria-label": "Event to tamper with", onchange: e => { S.selSeq = +e.target.value; S.confirmTamper = false; renderTx(); } }, opts);
  ctl.append(vbtn, res, h("span", { class: "spacer" }), sel,
    h("button", { class: "btn sm danger", type: "button", disabled: !ev.length, onclick: () => { S.confirmTamper = true; renderTx(); } }, "Tamper with event…"));
  const parts = [ctl];
  if (S.confirmTamper) {
    parts.push(h("div", { class: "confirm", role: "alertdialog" },
      `Overwrite the stored content of event #${S.selSeq}? The server's ledger stays broken from then on, for every transaction, until the server restarts.`,
      h("button", { class: "btn sm danger", type: "button", onclick: tamper }, `Overwrite event #${S.selSeq}`),
      h("button", { class: "btn sm", type: "button", onclick: () => { S.confirmTamper = false; renderTx(); } }, "Cancel")));
  }
  parts.push(h("div", { class: "chain", "aria-label": "Hash chain of this transaction" }, ev.map(e =>
    h("button", { type: "button", title: `${e.kind}\nprevious ${e.prev_hash}\nhash ${e.hash}`,
      class: "link" + (e.seq === bad ? " bad" : bad !== null && e.seq > bad ? " after" : "") + (e.seq === S.selSeq ? " sel" : ""),
      onclick: () => { S.selSeq = e.seq; S.confirmTamper = false; renderTx(); } },
      h("b", {}, "#" + e.seq), e.hash.slice(0, 6)))));
  return h("div", { class: "section" }, h("h2", {}, "Hash chain"),
    h("p", { class: "lede", style: "margin:0 0 .71rem" }, "Each event stores the hash of the one before it. Changing any stored event breaks the chain at that event."),
    h("div", { class: "panel" }, parts));
}
function defaultSeq(events) {
  const ev = events.find(e => e.kind === "evidence" && e.body && Array.isArray(e.body.observed) && !e.body.observed.length)
    || events.find(e => e.kind === "evidence") || events[0];
  return ev ? ev.seq : null;
}
function renderTx() {
  const root = $("#view-transaction");
  if (S.txLoading) {
    root.replaceChildren(h("div", { class: "head" }, h("div", {}, h("h1", {}, "Running transaction"),
      h("p", { class: "lede" }, "The agent is calling tools through the gateway."))), skeleton(), h("div", { style: "height:1rem" }), skeleton(["w80", "w80", "w60", "w80", "w60", "w30"]));
    return;
  }
  const parts = [];
  if (S.txErr) parts.push(errorBox(S.txErr, "The transaction could not be loaded"));
  const d = S.tx;
  if (!d) {
    const sc = scenario();
    parts.push(h("div", { class: "panel empty-state" }, h("h2", {}, "No transaction yet"),
      h("p", {}, "Pick a scenario, an agent, a gateway mode and a fault, then run it. The agent's calls, the gateway's ledger and the system of record appear here side by side."),
      sc ? h("p", { class: "muted" }, `Selected: ${sc.task_id}, requesting ${sc.requested.map(s => (STEP_NAME[s] || s).toLowerCase()).join(", ")}.`) : null));
    root.replaceChildren(...parts);
    return;
  }
  const injected = d.never_commit ? `${d.never_commit} never commits`
    : (d.faults || []).length ? d.faults.map(f => `${f.tool}, ${ORD[f.k] || "call " + (f.k + 1)} call, ${f.kind}`).join("; ") : "no fault";
  parts.push(h("div", { class: "head" }, h("div", { style: "min-width:0" },
    h("h1", {}, `${d.task_id}: ${d.agent || "external"} agent, ${MODE_NAME[d.mode] || d.mode}`),
    h("div", { class: "ids" }, `${d.tx_id}  ·  fault injected: ${injected}`))));
  parts.push(contractNode(d), outcomeNode(d), stepsNode(d), lanesNode(d, S.events), chainNode(d));
  root.replaceChildren(...parts);
}
async function loadTx(id) {
  S.txLoading = true; S.txErr = null; renderTx(); syncForm();
  try {
    const [d, ev] = await Promise.all([api("GET", `/v1/transactions/${encodeURIComponent(id)}`), api("GET", `/v1/transactions/${encodeURIComponent(id)}/events`)]);
    S.tx = d; S.events = ev; S.selSeq = defaultSeq(ev); S.confirmTamper = false;
  } catch (e) { S.txErr = e; S.tx = null; }
  S.txLoading = false; renderTx(); syncForm();
}
async function runTx() {
  S.txLoading = true; S.txErr = null; S.confirmTamper = false; renderTx(); syncForm();
  try {
    const d = await api("POST", "/v1/test/runs", formBody());
    const ev = await api("GET", `/v1/transactions/${d.tx_id}/events`);
    S.tx = d; S.events = ev; S.selSeq = defaultSeq(ev);
    if (S.verify && S.verify.ok) S.verify = null;
    history.replaceState(null, "", `${location.search}#/transaction/${d.tx_id}`);
  } catch (e) { S.txErr = e; }
  S.txLoading = false; renderTx(); syncForm();
}
async function verifyChain() {
  try { S.verify = await api("GET", "/v1/ledger/verify"); S.verifyErr = null; } catch (e) { S.verifyErr = e; }
  renderTop(); if (S.view === "transaction") renderTx();
}
async function tamper() {
  S.confirmTamper = false;
  try {
    S.verify = await api("POST", "/v1/test/tamper", { seq: S.selSeq });
    S.verifyErr = null;
    if (S.tx) S.events = await api("GET", `/v1/transactions/${S.tx.tx_id}/events`);
  } catch (e) { S.verifyErr = e; }
  renderTop(); renderTx();
}

/* ------------------------------------------------------------------ compare view */
function compareCaption(rows) {
  const by = Object.fromEntries(rows.map(r => [r.mode, r]));
  const say = (r, name) => {
    if (!r) return null;
    let s = `${name} ${r.verified_completion ? "completed the work" : "did not complete the work"}`;
    if (r.false_finalization) s += " yet finalized it as a success";
    else if (!r.verified_completion) s += ` and ended ${r.disposition}`;
    if (r.verified_completion && r.recovery_actions.length) s += ` using ${[...new Set(r.recovery_actions)].map(a => (ACTION_TEXT[a] || a).toLowerCase()).join(" and ")}`;
    if (r.duplicate_payment) s += ", paying twice";
    return s;
  };
  const p = [say(by.passthrough, "Without a gateway the run"), say(by.enforce, "under enforce it")].filter(Boolean);
  return p.join("; ") + ".";
}
function renderCompare() {
  const root = $("#view-compare");
  const head = h("div", { class: "head" }, h("div", {}, h("h1", {}, "Compare modes"),
    h("p", { class: "lede" }, "The same scenario, agent and fault, executed once under each gateway mode. Every row is a real run with its own ledger.")));
  if (S.compareLoading) { root.replaceChildren(head, skeleton(["w80", "w80", "w80", "w80"])); return; }
  const parts = [head];
  if (S.compareErr) parts.push(errorBox(S.compareErr, "The comparison could not be run"));
  const c = S.compare;
  if (!c) {
    if (!S.compareErr) parts.push(h("div", { class: "panel empty-state" }, h("h2", {}, "No comparison yet"),
      h("p", {}, "Choose the inputs on the left and press Compare all four modes. Four transactions are executed and summarised here.")));
    root.replaceChildren(...parts); return;
  }
  const flag = (v, bad) => h("span", { class: v ? (bad ? "yes" : "good") : "" }, v ? "Yes" : "No");
  const injected = c.faults === null || c.faults === undefined ? "the scenario's own fault" : c.faults.length ? c.faults.map(f => `${f.tool} ${ORD[f.k]} call ${f.kind}`).join("; ") : "no fault or a never-commit tool";
  parts.push(h("p", { class: "caption" }, compareCaption(c.rows)));
  parts.push(h("p", { class: "lede", style: "margin:-.57rem 0 1rem" }, h("span", { class: "mono" }, c.task_id), `, ${c.agent} agent, ${injected}.`));
  parts.push(h("div", { class: "panel tblwrap" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, ["Mode", "Verified completion", "False finalization", "Agent false claim", "Duplicate payment", "Disposition", "Money exposure", "Agent calls", "Recovery actions", "Transaction"]
      .map((t, i) => h("th", { class: [6, 7].includes(i) ? "num" : "" }, t)))),
    h("tbody", {}, c.rows.map(r => h("tr", {},
      h("td", {}, h("b", { style: "font-weight:500" }, MODE_NAME[r.mode] || r.mode)),
      h("td", {}, flag(r.verified_completion, false)),
      h("td", {}, flag(r.false_finalization, true)),
      h("td", {}, flag(r.agent_false_claim, true)),
      h("td", {}, flag(r.duplicate_payment, true)),
      h("td", { class: "mono" }, r.disposition === null ? "none" : r.disposition, r.mode === "passthrough" ? h("div", { class: "muted", style: "font-family:var(--sans);font-size:.79rem" }, "agent's own report") : null),
      h("td", { class: "num" }, r.exposure_cents === null || r.exposure_cents === undefined ? h("span", { class: "muted" }, "not measured") : money(r.exposure_cents)),
      h("td", { class: "num" }, int(r.agent_calls)),
      h("td", {}, r.recovery_actions.length ? r.recovery_actions.map(a => ACTION_TEXT[a] || a).join(", ") : h("span", { class: "muted" }, "none")),
      h("td", { class: "mono" }, h("a", { href: `#/transaction/${r.tx_id}` }, r.tx_id))))))));
  parts.push(h("p", { class: "lede", style: "margin-top:.86rem;font-size:.82rem" },
    "Verified completion, false finalization and duplicate payment come from the truth verifier, which reads the final state and shares no code with the gateway. Agent false claim compares the agent's own report with the same final state."));
  root.replaceChildren(...parts);
}
async function runCompare() {
  S.compareLoading = true; S.compareErr = null; renderCompare(); syncForm();
  try { S.compare = await api("POST", "/v1/test/compare", formBody()); } catch (e) { S.compareErr = e; }
  S.compareLoading = false; renderCompare(); syncForm();
}

/* ------------------------------------------------------------------ live ledger */
function setStream(status) { S.stream.status = status; renderTop(); if (S.view === "ledger") renderLedgerHead(); }
function connect() {
  const st = S.stream;
  if (st.es) st.es.close();
  setStream(st.last ? "reconnecting" : "connecting");
  const es = new EventSource(`/v1/stream?after=${st.last}`);
  st.es = es;
  es.onopen = async () => {
    st.backoff = 1000; setStream("live");
    if (!st.last) return;
    // a restarted gateway starts a new ledger at event 1; replaying after our old position would skip it
    try {
      const v = await api("GET", "/v1/ledger/verify");
      if (v.ok && v.n < st.last) { st.last = 0; st.events = []; st.seen.clear(); if (S.view === "ledger") fillLedger(); connect(); }
    } catch { /* the stream itself reports trouble */ }
  };
  es.onmessage = m => {
    let e;
    try { e = JSON.parse(m.data); } catch { return; }
    if (e.seq <= st.last) return;
    st.last = e.seq;
    st.events.push(e);
    if (st.events.length > 2000) st.events.splice(0, st.events.length - 2000);
    st.seen.add(e.tx_id);
    if (S.view === "ledger") queueLedgerRow(e);
    renderTop();
  };
  es.onerror = () => {
    es.close();
    st.reconnects++;
    setStream("reconnecting");
    st.backoff = Math.min((st.backoff || 1000) * 2, 10000);
    setTimeout(connect, st.backoff);
  };
}
let pending = [], flushTimer = null;
function queueLedgerRow(e) {
  pending.push(e);
  if (!flushTimer) flushTimer = setTimeout(() => {
    flushTimer = null;
    const body = $("#ledger-body");
    if (!body) { pending = []; return; }
    const rows = pending.filter(match).reverse().map(e => ledgerRow(e, true));
    pending = [];
    body.prepend(...rows);
    while (body.children.length > 1000) body.lastChild.remove();
    $("#ledger-empty").hidden = body.children.length > 0;
    renderLedgerHead();
    setTimeout(() => rows.forEach(r => r.classList.remove("fresh")), 900);
  }, 30);
}
const match = e => !S.stream.filter || e.tx_id.includes(S.stream.filter);
function ledgerRow(e, fresh) {
  const d = describe(e);
  return h("tr", { class: fresh ? "fresh" : "" },
    h("td", { class: "num" }, e.seq), h("td", { class: "mono" }, clock(e.t_wall)),
    h("td", { class: "mono" }, h("a", { href: `#/transaction/${e.tx_id}` }, e.tx_id)),
    h("td", {}, e.actor), h("td", { class: "mono" }, e.kind),
    h("td", { class: "sum" + (d.cls ? " pen" : "") }, d.what + (d.tag && !d.what.includes(d.tag[0]) ? ` (${d.tag[0]})` : "") + (d.det ? `. ${d.det}` : "")),
    h("td", { class: "mono", title: e.hash }, e.hash.slice(0, 10)));
}
function renderLedgerHead() {
  const el = $("#ledger-status");
  if (!el) return;
  const st = S.stream, shown = $("#ledger-body") ? $("#ledger-body").children.length : 0;
  const label = { live: "Live", reconnecting: "Reconnecting", connecting: "Connecting" }[st.status];
  el.replaceChildren(h("span", { class: "dot " + (st.status === "live" ? "live" : "") }), label);
  $("#ledger-count").textContent = `${int(st.events.length)} events received, last #${st.last}, ${int(shown)} shown${st.reconnects ? `, ${st.reconnects} reconnects replayed from the last event` : ""}`;
  const dl = $("#ledger-txs");
  if (dl && dl.children.length !== st.seen.size) dl.replaceChildren(...[...st.seen].reverse().map(t => h("option", { value: t })));
}
function renderLedger() {
  const root = $("#view-ledger");
  const input = h("input", { type: "search", placeholder: "Filter by transaction id", list: "ledger-txs", "aria-label": "Filter by transaction id", value: S.stream.filter,
    oninput: e => { S.stream.filter = e.target.value.trim(); fillLedger(); } });
  root.replaceChildren(
    h("div", { class: "head" }, h("div", {}, h("h1", {}, "Live ledger"),
      h("p", { class: "lede" }, "Every event appended to the evidence ledger, newest first, streamed from the gateway as it happens. After a dropped connection the stream resumes from the last event received."))),
    h("div", { class: "toolbar" }, h("span", { class: "status-pill", id: "ledger-status", role: "status" }), input, h("datalist", { id: "ledger-txs" }),
      h("span", { class: "muted", id: "ledger-count", style: "font-size:.86rem" })),
    h("div", { class: "panel tblwrap" }, h("table", { class: "tbl ledger" },
      h("thead", {}, h("tr", {}, ["Event", "Time", "Transaction", "Actor", "Kind", "Summary", "Hash"].map((t, i) => h("th", { class: i === 0 ? "num" : "" }, t)))),
      h("tbody", { id: "ledger-body" })),
      h("div", { class: "empty-state", id: "ledger-empty" }, h("h2", {}, "No events yet"),
        h("p", {}, "Run a transaction or a comparison and its ledger events appear here as they are written."))));
  fillLedger();
}
function fillLedger() {
  const body = $("#ledger-body");
  const rows = S.stream.events.filter(match).slice(-1000).reverse().map(e => ledgerRow(e, false));
  body.replaceChildren(...rows);
  $("#ledger-empty").hidden = rows.length > 0;
  if (!rows.length && S.stream.filter && S.stream.events.length) {
    $("#ledger-empty").replaceChildren(h("h2", {}, "No matching events"), h("p", {}, `No event in this session belongs to a transaction containing "${S.stream.filter}".`));
  }
  renderLedgerHead();
}

/* ------------------------------------------------------------------ evaluation */
const EVAL_META = {
  h7_matrix: ["Defences against execution faults", "Hypothesis 7. Each defence on identical fault traces in ProcureSim worlds. Rates are over episodes; exposure is mean cents paid beyond one verified payment."],
  reward_truth_table: ["Reward truth table", "Amendment 12 registration artefact, computed before training. Verdict reward (R_V) is 1 when every requested step holds in the final state; acknowledgement reward (R_A) is 1 when the last reply for every step said ok."],
  a12_gate: ["Amendment 12 gate", "Checks that must pass before reinforcement learning starts."],
  a12_h3_predictions: ["Amendment 12, hypothesis 3 predictions", "Predicted sign of the advantage of reading back, per reward and fault class."],
  a12_summary: ["Amendment 12 summary", ""],
  a11_summary: ["Amendment 11 summary", ""],
  h4_summary: ["Hypothesis 4 summary", ""],
  stage1_confirmatory: ["Stage 1 confirmatory results", "Pre-registered confirmatory analysis on the held-out test splits."],
};
function table(headers, rows, numCols) {
  return h("div", { class: "tblwrap" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, headers.map((t, i) => h("th", { class: (numCols || []).includes(i) ? "num" : "" }, t)))),
    h("tbody", {}, rows)));
}
function flatten(obj, prefix, out, depth) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v) && depth < 3) flatten(v, key, out, depth + 1);
    else out.push([key, v]);
  }
  return out;
}
function valText(v) {
  if (Array.isArray(v)) return v.every(x => x === null || typeof x !== "object") ? v.map(x => fmtNum(x)).join(", ") : JSON.stringify(v);
  if (v && typeof v === "object") return JSON.stringify(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  return fmtNum(v);
}
function kvTable(data) {
  const rows = Array.isArray(data) ? data.map((v, i) => [String(i), v]) : flatten(data, "", [], 0);
  return h("div", { class: "tblwrap" }, h("table", { class: "tbl kvt" }, h("tbody", {}, rows.map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, valText(v)))))));
}
function h7Node(data) {
  const faults = [...new Set(data.map(r => r.fault))];
  const cols = ["Defence", "Episodes", "Goal met", "Duplicate", "False finalization", "Honest disclosure", "Mean exposure (cents)", "Mean calls"];
  const rows = [];
  for (const f of faults) {
    rows.push(h("tr", { class: "group" }, h("td", { colspan: cols.length }, `Fault: ${f}`)));
    for (const r of data.filter(x => x.fault === f)) {
      rows.push(h("tr", {}, h("td", {}, r.defence), h("td", { class: "num" }, int(r.n)),
        h("td", { class: "num" }, h("span", { class: r.goal === 1 ? "good" : "" }, fmtNum(r.goal, 3))),
        h("td", { class: "num" }, h("span", { class: r.dup > 0 ? "yes" : "" }, fmtNum(r.dup, 3))),
        h("td", { class: "num" }, h("span", { class: r.false_final > 0 ? "yes" : "" }, fmtNum(r.false_final, 3))),
        h("td", { class: "num" }, fmtNum(r.disclosed, 3)),
        h("td", { class: "num" }, fmtNum(Math.round(r.exposure))), h("td", { class: "num" }, fmtNum(r.calls, 1))));
    }
  }
  return table(cols, rows, [1, 2, 3, 4, 5, 6, 7]);
}
function rewardNode(data) {
  const parts = [];
  if (Array.isArray(data.rows)) {
    const cols = ["Fault", "Behaviour", "Verdict reward (R_V)", "Acknowledgement reward (R_A)", "Calls", "Reads back", "Verified completion", "Payment verdict"];
    parts.push(table(cols, data.rows.map(r => h("tr", {}, r.map((c, i) => h("td", { class: i >= 2 && i <= 6 && i !== 5 ? "num" : i === 7 ? "mono" : "" }, String(c))))), [2, 3, 4, 6]));
  }
  if (Array.isArray(data.signs)) {
    parts.push(h("div", { class: "ecap" }, "Sign of the advantage of reading back: best reward over reading behaviours minus best reward over non-reading behaviours, per fault."));
    parts.push(table(["Fault", "Verdict reward (R_V)", "Acknowledgement reward (R_A)"], data.signs.map(r => h("tr", {}, r.map((c, i) => h("td", { class: i ? "mono" : "" }, String(c)))))));
  }
  return parts.length ? h("div", {}, parts) : kvTable(data);
}
function gateNode(data) {
  const parts = [];
  if (data.checks && typeof data.checks === "object") {
    parts.push(table(["Check", "Result"], Object.entries(data.checks).map(([k, v]) => h("tr", {}, h("td", { class: "mono" }, k),
      h("td", {}, h("span", { class: "tag " + (v ? "ok" : "bad") }, v ? "pass" : "fail"))))));
  }
  const rest = Object.fromEntries(Object.entries(data).filter(([k]) => k !== "checks"));
  parts.push(kvTable(rest));
  return h("div", {}, parts);
}
function renderEvaluation() {
  const root = $("#view-evaluation");
  const head = h("div", { class: "head" }, h("div", {}, h("h1", {}, "Evaluation"),
    h("p", { class: "lede" }, "Result files behind the claims, read from the repository as they are. A file that does not exist yet is shown as pending, never estimated.")));
  if (S.evalErr) { root.replaceChildren(head, errorBox(S.evalErr, "Evaluation files could not be loaded")); return; }
  if (!S.evaluation) { root.replaceChildren(head, skeleton(), h("div", { style: "height:1rem" }), skeleton()); return; }
  const secs = Object.entries(S.evaluation).map(([k, v]) => {
    const [title, cap] = EVAL_META[k] || [human(k), ""];
    let body;
    if (v.status !== "available") body = h("div", { class: "pending" }, "Pending. This file has not been produced yet, so no numbers are shown.");
    else {
      try {
        const d = v.data;
        body = k === "h7_matrix" && Array.isArray(d) ? h7Node(d)
          : k === "reward_truth_table" && d && typeof d === "object" ? rewardNode(d)
          : k === "a12_gate" && d && typeof d === "object" ? gateNode(d)
          : d && typeof d === "object" ? kvTable(d) : h("div", { class: "pending mono" }, valText(d));
      } catch (e) { body = errorBox(e, "This file has an unexpected shape"); }
    }
    return h("section", { class: "panel evalsec" },
      h("div", { class: "eh" }, h("h2", {}, title), h("span", { class: "path" }, v.path),
        h("span", { class: "tag " + (v.status === "available" ? "ok" : "") }, v.status === "available" ? "available" : "pending")),
      cap && v.status === "available" ? h("div", { class: "ecap" }, cap) : null,
      h("div", { class: "eb" }, body));
  });
  const avail = Object.values(S.evaluation).filter(v => v.status === "available").length;
  root.replaceChildren(head, h("p", { class: "lede", style: "margin:-.43rem 0 1rem" }, `${avail} of ${Object.keys(S.evaluation).length} files available.`), ...secs);
}
async function loadEvaluation() {
  S.evalErr = null; S.evaluation = null; renderEvaluation();
  try { S.evaluation = await api("GET", "/v1/evaluation"); } catch (e) { S.evalErr = e; }
  renderEvaluation();
}

/* ------------------------------------------------------------------ top bar and routing */
function renderTop() {
  const st = S.stream;
  const label = { live: "Ledger stream live", reconnecting: "Ledger stream reconnecting", connecting: "Ledger stream connecting" }[st.status];
  $("#top-stream").replaceChildren(h("span", { class: "dot " + (st.status === "live" ? "live" : "") }), label,
    st.last ? h("span", { class: "mono muted", style: "margin-left:.43rem" }, `#${st.last}`) : null);
  const v = S.verify;
  $("#top-chain").replaceChildren(!v ? "Chain not checked yet"
    : v.ok ? h("span", {}, h("span", { class: "dot live" }), `Chain intact at ${int(v.n)} events`)
      : h("span", { class: "pen" }, h("span", { class: "dot bad" }), `Chain broken at event #${v.first_bad_seq}`));
}
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  const view = ["transaction", "compare", "ledger", "evaluation"].includes(parts[0]) ? parts[0] : "transaction";
  S.view = view;
  document.querySelectorAll(".nav a").forEach(a => {
    if (a.dataset.view === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  document.querySelectorAll("#main > section").forEach(s => { s.hidden = s.dataset.view !== view; });
  const withForm = view === "transaction" || view === "compare";
  $("#side").hidden = !withForm;
  $("#page").classList.toggle("wide", !withForm);
  syncForm();
  if (view === "transaction") {
    const id = parts[1] ? decodeURIComponent(parts[1]) : null;
    if (id && (!S.tx || S.tx.tx_id !== id) && !S.txLoading) loadTx(id); else renderTx();
  } else if (view === "compare") renderCompare();
  else if (view === "ledger") renderLedger();
  else loadEvaluation();
}
window.addEventListener("hashchange", route);
buildForm();
renderTop();
route();
loadScenarios();
connect();

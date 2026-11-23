// ROI projection, the same formula as scripts/roi_model.py `project` (tests/product/test_site_upgrade.py checks they agree).
// arm: {rates, calls, n, llm} measured in ProcureSim; s: {fault_rate, value, exception_cost, late_multiplier,
// dup_recovery, gpu_hour} assumed; epgh: measured policy calls per GPU-hour.
function apRoiProject(arm, s, epgh) {
  const CLASSES = ["clean", "phantom", "partial", "dup_delivery", "lost_ack", "precommit_timeout", "never_commit"];
  const faulty = CLASSES.filter(c => c !== "clean" && arm.rates[c]);
  const tot = faulty.reduce((a, c) => a + arm.n[c], 0);
  const w = Object.fromEntries(faulty.map(c => [c, arm.n[c] / tot]));
  const p = o => (1 - s.fault_rate) * ((arm.rates.clean || {})[o] || 0) +
    s.fault_rate * faulty.reduce((a, c) => a + w[c] * (arm.rates[c][o] || 0), 0);
  const calls = (1 - s.fault_rate) * (arm.calls.clean || 0) + s.fault_rate * faulty.reduce((a, c) => a + w[c] * arm.calls[c], 0);
  const per = {};
  for (const o of ["verified", "false_success", "unrefunded_duplicate", "flagged"]) per[o] = 1000 * p(o);
  const cost = {
    false_success_usd: per.false_success * s.exception_cost * s.late_multiplier,
    duplicate_usd: per.unrefunded_duplicate * (s.value * (1 - s.dup_recovery) + s.exception_cost * s.late_multiplier),
    flagged_usd: per.flagged * s.exception_cost,
    inference_usd: arm.llm ? 1000 * calls / epgh * s.gpu_hour : 0,
  };
  cost.total_usd = cost.false_success_usd + cost.duplicate_usd + cost.flagged_usd + cost.inference_usd;
  return { per_1000_ops: per, calls_per_op: calls, cost_per_1000_ops: cost };
}
if (typeof module !== "undefined") module.exports = { apRoiProject };

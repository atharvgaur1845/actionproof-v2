# Video command sheet

Driver: Claude. Run from the repository root after `pip install -e ".[integrity]"`. Each command below was run on 2026-11-20 with the outputs described (scripted `naive` agent, task `proc_dev-0001`). Re-run each one before recording; if an output differs, the script segment changes, not the command.

## Setup (before recording)

```
actionproof serve --port 8765        # dashboard: http://127.0.0.1:8765/
```

## 1:30 hero run: phantom payment

```
actionproof compare --task proc_dev-0001 --agent naive --fault pay_supplier:0=phantom
```

| Mode | Verified completion | False finalization | Disposition | Recovery |
|---|---|---|---|---|
| passthrough | 0 | 1 | completed | none |
| observe | 0 | 0 | failed | none |
| enforce | 1 | 0 | completed | SAFE_RETRY |
| enforce_inform | 1 | 0 | completed | SAFE_RETRY |

## 3:00 hard cases

```
actionproof compare --task proc_dev-0001 --agent naive --fault pay_supplier:0=never_commit
```
Passthrough finalizes a false success; Enforce retries twice with the same key, then escalates. Nothing is paid.

```
actionproof compare --task proc_dev-0001 --agent naive --fault pay_supplier:0=dup_delivery
```
Passthrough finalizes with two payments; Observe reports the duplicate with its exposure; Enforce refunds the extra payment (COMPENSATE) and completes.

```
actionproof compare --task proc_dev-0001 --agent naive --fault pay_supplier:0=partial
```
Enforce refunds the partial payment, then pays the full amount (COMPENSATE, SAFE_RETRY).

## Tamper detection

In the dashboard, open a finished transaction, pick an event under "Hash chain", press "Tamper with event…" and confirm: the chain shows "Chain broken at event #N". To check a stored ledger from the shell, start the server with a file-backed ledger:

```
actionproof serve --port 8765 --ledger ledger.db
actionproof ledger verify --ledger ledger.db
```

## Contract

```
actionproof contract show --task proc_dev-0001
```

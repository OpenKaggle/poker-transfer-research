# EVP-DRAFT-04.2 — canonical-root and consumption-integrity delta

**Status:** `IMPLEMENTATION_DELTA_ONLY_AWAITING_FRESH_SOURCE_AUDIT`  
**Date:** 2026-09-11  
**Authority:** versioned local repair only. No development-data access, model
fit, aggregate evaluation, Kaggle action, experiment number, candidate
promotion, submission, or rerun is authorized.

This delta supplements EVP-DRAFT-04 and the v4.1 implementation delta. It does
not overwrite or rehabilitate any v4 or v4.1 audit object. All statistical
methods and every v4.1 repair remain frozen. V4.2 changes only the three source
integrity controls below.

## Immutable v4.1 binding

| Artifact | SHA-256 |
|---|---|
| v4.1 delta | `285a2b85beb03abf4419c4e3816bd3c0f0af68ebb64eec4c01a1fed9a3db4279` |
| v4.1 core | `d20d2b3c5fd537cd827e2817c911860d605687bc84eb33bfcce2c9340534b67c` |
| v4.1 runner | `8deb18fe3f688d2bc4c02f7e7a25ae97acfc9558ba4f3001853b7f0af4ab9175` |
| v4.1 core tests | `96ff3061d3668cd7d9b6c5b17ce5a419d2fe459f6e83ab7dd1df04b7b6df33c7` |
| v4.1 runner tests | `1b8252f486cdfa839b9a79399200f5a07f81de562d0ea282fd9077ffa1386905` |
| v4.1 static audit | `2148b28d4837b06bfeb1d9ea0fa4f11abb9fae2ced41289af5619ffd275dabef` |
| v4.1 containment | `144800fbb7bda2c7f3dc5bd52e6f4fa3a43834815bb302ed3561e319108d3adb` |

## Mandatory v4.2 repairs

### 1. Canonical result root and one global consumption point

The result root is exactly the executor-constructed canonical directory
`poker/work`. Its identity is established by the canonical, immutable
`poker/work/EVP_v4_2_result_root_binding.json` and that file's exact SHA-256.
A future result authorization must bind both the canonical relative root and
the marker path/hash. The executor rejects a caller-selected alternate,
relative, symlinked, noncanonical, or external result root before authorization
consumption. All consumption and result artifacts are published only inside
this one canonical root, so the same authorization cannot be replayed through a
different directory.

### 2. Durable fail-closed consumption and exception terminalization

Consumption uses two append-only artifacts in the canonical root. First, an
authoritative claim is reserved with `O_CREAT | O_EXCL`; claim existence alone
means the authorization is consumed, including after a zero-byte or partial
write. Second, a complete hash-bound JSON receipt is prepared, flushed, and
published atomically without overwrite. A failure at any point after claim
creation remains consumed. The exception handler must not require a valid
consumption JSON: it records claim/receipt presence and available hashes in the
canonical append-only terminal STOP journal and re-raises. A later attempt
stops on either pre-existing consumption artifact.

### 3. Exact Stage-2 terminal controls

Every terminal journal adds explicit booleans named
`stage_2_development_data_access`, `stage_2_model_fit`, and
`stage_2_aggregate_performance_viewed`, in addition to the detailed v4.1
controls. They become true exactly when Stage 2 uses the in-memory confirmed
development universe, completes a matched-C L0 model fit, and computes/views
its aggregate attribution result respectively.

## Verification and next gate

Implementation verification is restricted to source-static checks and pure
synthetic fault injection covering alternate-root replay, symlink roots,
partial consumption writes, and exact Stage-2 controls. V4.2 cannot issue its
own independent source-audit PASS. A separately authorized fresh source audit
must bind the exact v4.2 delta, root marker, core, runner, tests, static audit,
and containment hashes before any result-bearing authorization can be
considered.

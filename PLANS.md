# Project plan

## Current release objective

Make MLX Large Model Lab safe to clone, install, and operate without mixing a
contributor's hardware, model downloads, logs, or benchmark history into the
Git repository.

## Shipped

- Immutable packaged model capability catalog.
- Schema-versioned per-user configuration with hardware and model-path state.
- User-store directories for weights, cache, runs, logs, telemetry, and server
  runtime state.
- Legacy checkout model discovery without copying large artifacts.
- Shared resolution across CLI, server wrapper, shell launcher, and Control
  Center.
- Pre-download model fit planning with exact or offline size estimates,
  context-sensitive KV-cache math, disk checks, and adaptive memory reserves.
- Advisory agent-loop detection using sustained tool chains, context growth,
  prompt-cache loss, and latency degradation, surfaced in Control Center
  telemetry without automatic cancellation.
- Server-paginated recent-query telemetry with daily JSONL archives and an
  automatic 30-day deletion cutoff.
- Classified telemetry failures with selectable chart buckets, exact
  client/endpoint/model details, and service restart controls.
- Contributor, security, configuration, implementation-status, and audit docs.

## Public-release blockers

1. Repository owner selects and adds an OSI-approved license.
2. Add a macOS CI workflow for unit tests, shell validation, and wheel-content
   verification.
3. Verify a clean clone on a second Apple Silicon machine with no legacy
   checkout state.

## Next engineering slices

1. Package the server lifecycle helper as a Python entry point so installed
   wheels do not depend on checkout-relative shell scripts.
2. Add explicit user-config schema migrations before a version 2 change.
3. Add CLI commands for selecting the preferred default model and removing a
   stale registered path without hand-editing JSON.
4. Add import/export diagnostics that exclude private paths by default.
5. Surface model-fit plans and smaller-model recommendations directly in the
   Control Center.

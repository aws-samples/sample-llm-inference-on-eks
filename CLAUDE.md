# llm-inference-on-eks — Working Conventions

Guidance for AI assistants (and humans) working in this repo. Personal
environment values (account ID, region, capacity details) live in the
git-ignored `CLAUDE.local.md` — never in committed files.

## Placeholder Convention (IMPORTANT)

This repo is published as a public sample. Committed manifests/docs use
`<ACCOUNT_ID>` and `<REGION>` placeholders for anything account-specific
(ECR image URIs etc.).

- NEVER commit real account IDs, capacity-reservation IDs, or other
  environment-specific identifiers.
- NEVER `kubectl apply` a file containing literal placeholders — substitute
  first (real values in `CLAUDE.local.md`):

  ```bash
  sed "s|<ACCOUNT_ID>|$ACCOUNT_ID|g; s|<REGION>|$REGION|g" <file>.yaml | kubectl apply -f -
  ```

- `local/` (git-ignored) may hold pre-rendered real-value copies. It does not
  auto-sync with source manifests; prefer on-the-fly substitution.
- Before committing, scan for leaked identifiers:

  ```bash
  grep -rnE '[0-9]{12}\.dkr\.ecr|cr-[0-9a-f]{17}' --include='*.yaml' --include='*.md' --include='Dockerfile*' . | grep -v '^./local/'
  ```

## Repo Conventions

- Manifest naming: `<model>-<engine>[-<instance>].yaml`; LWS:
  `lws-<model>-<topology>-<instance>.yaml`; Dockerfiles:
  `Dockerfile.efa[-nixl]-sglang-<version>`
- New manifest → add a row to `docs/MODEL-INDEX.md` (a PostToolUse hook in
  `.claude/settings.json` reminds about this) and keep the README layout
  section current
- Pin image versions — no `:latest`
- No hardcoded `namespace:` in manifests — deploy to the current context
- `privileged: true` only in `lws/` manifests (EFA device access); single-node
  manifests must not need it
- Old manifests are kept as reference (📦 in `docs/MODEL-INDEX.md`); review
  before applying

## Useful Context

- Model weights cache on node-local NVMe at `/mnt/k8s-disks/0/models/<family>`
  (EKS GPU AMI instance-store RAID); first pull of a ~750 GB model takes
  20–40 min, restarts on the same node are fast
- Benchmark workflow, tool-version gotchas, and thinking-model pitfalls:
  `docs/benchmark-commands.md` and `docs/GLM-5.2-BENCHMARK.md`

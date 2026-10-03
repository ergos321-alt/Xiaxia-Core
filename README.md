# Xiaxia Core — production source recovery baseline

This repository is an archival recovery of the Xiaxia Core application source captured from the running production image on 2026-10-03. The container image is the source authority for this snapshot. No application behavior was changed during recovery.

## Production provenance

- Container: `xiaxia-core-xiaxia-core-1`
- Image tag: `xiaxia-core:browser-capabilities-action4r1-20260926`
- Image digest: `sha256:961afcc888c75484625b433de8f2f45a0c923e6598c55f41807ee383d4cff81c`
- Container working directory: `/app`
- Startup command: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log`
- Compose project/service: `xiaxia-core` / `xiaxia-core`
- The running container was managed by the `xiaxia-core` Compose project. Deployment files are kept outside this source-only baseline.
- Application source captured from `/app/app`, `/app/migrations`, and the top-level runtime files and dependency manifest in `/app`.

The host source directory `/opt/xiaxia/core/xiaxia-core` is not a Git repository and its application files differ from the running image. The repository contains the files captured from the image, not a merge with that older host tree.

## Snapshot contents and exclusions

The snapshot contains 53 files: 49 Python source files, three SQL migration files, and the production `requirements.txt`. `PRODUCTION_SOURCE_MANIFEST.sha256` records hashes for those captured files.

Excluded items were not needed as source code or contain runtime/private material: mounted `/app/data`, identity prompt content under `/app/identity`, `/opt/xiaxia-hermes-home` session state, `/run/secrets`, environment values, `.env`, logs, Python bytecode/cache, the installed virtual environment, and unimported `.pre-*` backup snapshots under `app/`. The production image contained no test tree. Host-only tests, docs, Dockerfile, and Compose files were not mixed into this source snapshot because they are absent from the image or do not represent the complete production source baseline.

The upstream Hermes runtime is an external image dependency at commit `f5d192611032025d2757b07ad838921872126182`; its third-party source is not vendored here. Xiaxia's integration and provider/tool wiring are included in `app/hermes_chat.py` and the other application modules.

Relevant environment configuration is server-side. The image exposes names including `MODEL_PROVIDER`, `QWEN_MODEL`, `QWEN_BASE_URL`, `BROWSER_TOOLS_ENABLED`, `BROWSER_NATIVE_TOOL_CALLING_ENABLED`, `CONTINUITY_ENABLED`, `CONTINUITY_SOURCE`, `MEMORY_ENABLED`, `IDENTITY_DIR`, and `DATABASE_PATH`. Secret values are intentionally absent from this repository.

## Runtime mapping observed in the image

- `app/api/routes.py`: App-facing HTTP routes, including `POST /v1/chat` and proactive conversation APIs.
- `app/hermes_chat.py`: Hermes `AIAgent` invocation, Qwen-compatible provider handoff, tool registration, Hermes-owned tool loop, tool-result handling, and Life cognition/delivery bridge.
- `app/config.py` and `app/models/qwen.py`: server-side provider/model configuration boundary.
- `app/main.py` and `app/life_runtime/`: Life Runtime initialization, scheduler lifecycle, persistence, and proactive delivery wiring.
- `app/core/native_browser.py`, `app/tools/browser.py`, and `app/browser/client.py`: Hermes Browser capability, available actions, Core client, and cloud tunnel endpoint.
- `app/identity/`, `app/memory/`, and `app/continuity/`: Identity loading, Memory interface, and continuity persistence/retrieval boundaries.
- `app/core/runtime.py`, `app/core/digital_identity_gateway.py`, `app/execution/`, and `app/tools/`: existing runtime and compatibility code preserved as captured.

This is a source snapshot, not an architectural reassignment: the repository records what the image contains and does not change ownership of semantic decisions, tools, or independent capabilities.

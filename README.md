# deepiri-ollama-utils

Python utilities for checking, managing, and selecting models on an Ollama server.

The package keeps Ollama HTTP operations separate from hardware sizing. GPU,
VRAM, model-fit, workload, and capacity decisions come directly from the
`deepiri-gpu-utils` Python library; this project never shells out to the
`deepiri-gpu` CLI.

## Requirements

- Python 3.11 or newer
- Ollama for commands that inspect or modify installed models
- `deepiri-gpu-utils` for recommendation and sizing commands

Python 3.11 is required because the inspected `deepiri-gpu-utils` package
requires Python 3.11. Until a tagged release containing the model-management
APIs is confirmed, `pyproject.toml` pins the exact inspected upstream commit.

For development with sibling checkouts:

```bash
python -m pip install -e ../deepiri-gpu-utils
python -m pip install -e . --no-deps
python -m pip install "httpx>=0.27" "pytest>=8"
```

The `--no-deps` development form keeps the editable sibling installation in
place. A normal installation resolves the pinned Git dependency.

## Ollama server URL

Commands use the following precedence:

1. `--base-url`
2. `OLLAMA_BASE_URL`
3. `http://localhost:11434`

For example:

```bash
export OLLAMA_BASE_URL=http://ollama:11434
deepiri-ollama-utils check
```

## Existing readiness commands

These commands remain backward-compatible and report JSON:

```bash
deepiri-ollama-utils check
deepiri-ollama-utils models
deepiri-ollama-utils has-model --model mistral:7b
deepiri-ollama-utils verify-models --model mistral:7b --model llama3:8b
deepiri-ollama-utils verify-models-file --file required-models.txt
```

They check whether Ollama is reachable and which models are already installed.
They do not install drivers, configure Docker, or alter models.

## GPU-aware model management

### Recommend models

```bash
deepiri-ollama-utils recommend-models
deepiri-ollama-utils recommend-models --backend-hint cuda --json
```

This combines `deepiri_ollama.tiers` recommendation helpers with the models
currently installed in Ollama. Hardware detection uses `deepiri-gpu-utils`.

### Inspect fit, workload, and capacity

```bash
deepiri-ollama-utils model-fit --model mistral:7b
deepiri-ollama-utils model-matrix
deepiri-ollama-utils workload --model mistral:7b --context-tokens 8192
deepiri-ollama-utils capacity --model mistral:7b --reserved-gb 2
```

These commands use local sizing modules backed by `deepiri-gpu-utils` hardware
detection:

- `deepiri_ollama.model_fit.model_fit_check`
- `deepiri_ollama.model_matrix.model_fit_matrix`
- `deepiri_ollama.workload.estimate_workload`
- `deepiri_ollama.capacity.model_capacity`

The estimates are advisory. Quantization, context length, batching, and current
GPU memory use can change actual Ollama requirements.

### Install and remove models

```bash
deepiri-ollama-utils install-model --model mistral:7b
deepiri-ollama-utils install-model --model mistral:7b --json
deepiri-ollama-utils remove-model --model mistral:7b
```

Installation:

1. obtains fit, workload, and capacity results from `deepiri-gpu-utils`;
2. displays any fit warning without inventing local sizing rules;
3. pulls through Ollama's `/api/pull` endpoint;
4. re-lists models and verifies that the model is installed.

Removal uses Ollama's `/api/delete` endpoint and verifies that the model no
longer appears in `/api/tags`.

## Python API

`deepiri_ollama.client.OllamaClient` provides reusable asynchronous operations:

- `list_models`
- `has_model`
- `show_model`
- `pull_model`
- `delete_model`
- `readiness`

`deepiri_ollama.model_management` combines those operations with the
gpu-utils recommendation, fit, workload, and capacity primitives.

## Cyrex workflow migration

`scripts/check-ollama-models.sh` is a thin compatibility entry point for the
reusable portion of the former
`diri-cyrex/scripts/llm/check-ollama-models.sh` workflow. It delegates to:

```bash
deepiri-ollama-utils recommend-models
```

The Python commands now replace Cyrex's reusable recommendation display, fit
warnings, installed-model checks, pull/install verification, and model removal.

The following intentionally remain outside this repository:

- GPU driver and NVIDIA Container Toolkit installation
- `sudo` operations and Docker daemon restarts
- Cyrex container names and Compose discovery
- Cyrex-to-Ollama network checks
- Cyrex provider/API-key and LangChain policy
- Helox, MLflow, S3, Redis, and model-registry workflows

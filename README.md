# deepiri-ollama-utils

CLI tool to check local Ollama availability and installed model names, with thin
delegation to `deepiri-gpu` for hardware and model-selection decisions.

## Local Ollama commands

- check
- models
- has-model
- verify-models
- verify-models-file

These commands query the local Ollama API. They answer whether Ollama is running
and whether exact model names are installed locally; they do not determine
whether a model fits the available hardware.

## Delegated GPU commands

- `model-matrix` -> `deepiri-gpu model-matrix`
- `capacity` -> `deepiri-gpu capacity`
- `workload` -> `deepiri-gpu workload`
- `model-fit` -> `deepiri-gpu model-fit`

All arguments after the command are passed directly to `deepiri-gpu`. This
project does not implement GPU detection, model sizing, capacity checks,
workload recommendations, or model-fit heuristics locally.

`deepiri-gpu` must be installed and available on `PATH` to use these delegated
commands. It is not currently declared as a package dependency of this project.

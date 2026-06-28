import argparse
import asyncio
import json
import subprocess
import sys
from .runtime import check, has_model, is_ollama_running, verify_models


DELEGATED_COMMANDS = ("model-matrix", "capacity", "workload", "model-fit")


def _delegate_to_deepiri_gpu(args):
    try:
        result = subprocess.run(["deepiri-gpu", *args], check=False)
    except FileNotFoundError:
        print(
            "deepiri-gpu was not found. Install deepiri-gpu-utils and ensure "
            "the deepiri-gpu executable is on PATH.",
            file=sys.stderr,
        )
        return 127

    return result.returncode


def _load_model_names_from_file(file_path):
    """Load model names from file, skipping blanks and comments."""
    models = []

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            models.append(name)

    return models


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)

    if raw_args and raw_args[0] in DELEGATED_COMMANDS:
        return _delegate_to_deepiri_gpu(raw_args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "check",
            "models",
            "has-model",
            "verify-models",
            "verify-models-file",
            *DELEGATED_COMMANDS,
        ],
    )
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", action="append", help="Model name to check")
    parser.add_argument("--file", help="File with one required model per line")

    args = parser.parse_args(raw_args)

    if args.command == "check":
        result = asyncio.run(check(args.base_url))
        print(json.dumps(result, indent=2))

    elif args.command == "models":
        result = asyncio.run(check(args.base_url))
        print(json.dumps(result, indent=2))

    elif args.command == "has-model":
        if not args.model:
            print(json.dumps({
                "ok": False,
                "base_url": args.base_url,
                "message": "Missing --model"
            }, indent=2))
            return

        model_name = args.model[0]
        running = asyncio.run(is_ollama_running(args.base_url))

        if not running:
            print(json.dumps({
                "ok": False,
                "running": False,
                "base_url": args.base_url,
                "model": model_name,
                "exists": False,
                "message": "Ollama not running"
            }, indent=2))
            return

        exists = asyncio.run(has_model(model_name, args.base_url))

        print(json.dumps({
            "ok": True,
            "running": True,
            "base_url": args.base_url,
            "model": model_name,
            "exists": exists,
            "message": "Model found" if exists else "Model not found"
        }, indent=2))

    elif args.command == "verify-models":
        if not args.model:
            print(json.dumps({
                "ok": False,
                "running": False,
                "base_url": args.base_url,
                "requested": [],
                "available": [],
                "missing": [],
                "all_present": False,
                "message": "Missing --model"
            }, indent=2))
            return

        result = asyncio.run(verify_models(args.model, args.base_url))
        print(json.dumps(result, indent=2))

    elif args.command == "verify-models-file":
        if not args.file:
            print(json.dumps({
                "ok": False,
                "running": False,
                "base_url": args.base_url,
                "requested": [],
                "available": [],
                "missing": [],
                "all_present": False,
                "message": "Missing --file argument"
            }, indent=2))
            return

        try:
            model_names = _load_model_names_from_file(args.file)
        except OSError as exc:
            print(json.dumps({
                "ok": False,
                "running": False,
                "base_url": args.base_url,
                "requested": [],
                "available": [],
                "missing": [],
                "all_present": False,
                "message": str(exc)
            }, indent=2))
            return

        if not model_names:
            print(json.dumps({
                "ok": False,
                "running": False,
                "base_url": args.base_url,
                "requested": [],
                "available": [],
                "missing": [],
                "all_present": False,
                "message": "No models found in file"
            }, indent=2))
            return

        result = asyncio.run(verify_models(model_names, args.base_url))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
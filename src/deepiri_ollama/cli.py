"""Command-line interface for Ollama readiness and model management."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from .client import OllamaClient, resolve_base_url
from .runtime import check, has_model, is_ollama_running, verify_models

EXISTING_COMMANDS = (
    "check",
    "models",
    "has-model",
    "verify-models",
    "verify-models-file",
)
MANAGEMENT_COMMANDS = (
    "recommend-models",
    "install-model",
    "remove-model",
    "model-fit",
    "model-matrix",
    "workload",
    "capacity",
)


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


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _emit(value: Any) -> None:
    print(json.dumps(_jsonable(value), indent=2, sort_keys=True))


def _single_model(args: argparse.Namespace) -> str | None:
    if not args.model:
        return None
    return args.model[0].strip() or None


def _missing_model(args: argparse.Namespace) -> int:
    _emit(
        {
            "ok": False,
            "base_url": args.base_url,
            "message": "Missing --model",
        }
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepiri-ollama-utils")
    parser.add_argument("command", choices=[*EXISTING_COMMANDS, *MANAGEMENT_COMMANDS])
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--model", action="append", help="Ollama model name")
    parser.add_argument("--file", help="File with one required model per line")
    parser.add_argument("--backend-hint", default=None, help="GPU backend hint for sizing")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=4096,
        help="Context window used by workload and capacity estimates",
    )
    parser.add_argument(
        "--reserved-gb",
        type=float,
        default=1.0,
        help="Memory reserved for the OS/runtime by capacity estimates",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON for management commands",
    )
    return parser


def _run_existing(args: argparse.Namespace) -> int:
    if args.command == "check":
        _emit(asyncio.run(check(args.base_url)))
        return 0

    if args.command == "models":
        _emit(asyncio.run(check(args.base_url)))
        return 0

    if args.command == "has-model":
        model_name = _single_model(args)
        if not model_name:
            _emit(
                {
                    "ok": False,
                    "base_url": args.base_url,
                    "message": "Missing --model",
                }
            )
            return 0

        running = asyncio.run(is_ollama_running(args.base_url))
        if not running:
            _emit(
                {
                    "ok": False,
                    "running": False,
                    "base_url": args.base_url,
                    "model": model_name,
                    "exists": False,
                    "message": "Ollama not running",
                }
            )
            return 0

        exists = asyncio.run(has_model(model_name, args.base_url))
        _emit(
            {
                "ok": True,
                "running": True,
                "base_url": args.base_url,
                "model": model_name,
                "exists": exists,
                "message": "Model found" if exists else "Model not found",
            }
        )
        return 0

    if args.command == "verify-models":
        if not args.model:
            _emit(
                {
                    "ok": False,
                    "running": False,
                    "base_url": args.base_url,
                    "requested": [],
                    "available": [],
                    "missing": [],
                    "all_present": False,
                    "message": "Missing --model",
                }
            )
            return 0
        _emit(asyncio.run(verify_models(args.model, args.base_url)))
        return 0

    if args.command == "verify-models-file":
        if not args.file:
            _emit(
                {
                    "ok": False,
                    "running": False,
                    "base_url": args.base_url,
                    "requested": [],
                    "available": [],
                    "missing": [],
                    "all_present": False,
                    "message": "Missing --file argument",
                }
            )
            return 0

        try:
            model_names = _load_model_names_from_file(args.file)
        except OSError as exc:
            _emit(
                {
                    "ok": False,
                    "running": False,
                    "base_url": args.base_url,
                    "requested": [],
                    "available": [],
                    "missing": [],
                    "all_present": False,
                    "message": str(exc),
                }
            )
            return 0

        if not model_names:
            _emit(
                {
                    "ok": False,
                    "running": False,
                    "base_url": args.base_url,
                    "requested": [],
                    "available": [],
                    "missing": [],
                    "all_present": False,
                    "message": "No models found in file",
                }
            )
            return 0

        _emit(asyncio.run(verify_models(model_names, args.base_url)))
        return 0

    raise AssertionError(f"Unhandled existing command: {args.command}")


def _render_recommendations(report: Any) -> None:
    print(
        f"Default model: {report.default_model} "
        f"(tier={report.setup_tier}, RAM={report.system_ram_gb}GB, "
        f"VRAM={report.effective_vram_gb}GB)"
    )
    if not report.ready:
        print(f"Ollama readiness: unavailable at {report.base_url}")
        if report.readiness_error:
            print(f"  {report.readiness_error}")

    for category in ("recommended", "usable", "marginal", "no"):
        rows = [row for row in report.rows if row.fit == category]
        if not rows:
            continue
        print(f"\n{category.upper()}:")
        for row in rows:
            marker = " [INSTALLED]" if row.installed else ""
            description = f" - {row.description}" if row.description else ""
            print(f"  {row.model}{marker}{description}")


async def _recommend(args: argparse.Namespace) -> Any:
    from .model_management import recommendation_report

    async with OllamaClient(args.base_url) as client:
        return await recommendation_report(client, backend_hint=args.backend_hint)


async def _install(args: argparse.Namespace, model: str) -> Any:
    from .model_management import install_model

    async with OllamaClient(args.base_url) as client:
        return await install_model(
            client,
            model,
            backend_hint=args.backend_hint,
            context_tokens=args.context_tokens,
            reserved_gb=args.reserved_gb,
        )


async def _remove(args: argparse.Namespace, model: str) -> Any:
    from .model_management import remove_model

    async with OllamaClient(args.base_url) as client:
        return await remove_model(client, model)


def _run_management(args: argparse.Namespace) -> int:
    try:
        if args.command == "recommend-models":
            report = asyncio.run(_recommend(args))
            if args.json:
                _emit(report)
            else:
                _render_recommendations(report)
            return 0

        if args.command == "model-matrix":
            from .model_matrix import model_fit_matrix, render_model_matrix_text

            matrix = model_fit_matrix()
            if args.json:
                _emit(matrix)
            else:
                print(render_model_matrix_text(matrix))
            return 0

        model = _single_model(args)
        if not model:
            return _missing_model(args)

        if args.command == "install-model":
            result = asyncio.run(_install(args, model))
            if args.json:
                _emit(result)
            else:
                if result.assessment.warning:
                    print(f"Warning: {result.assessment.warning}")
                print(
                    f"install-model: {result.model} -> "
                    f"{'installed' if result.success else 'failed'} "
                    f"(verified={result.verified})"
                )
                if result.error:
                    print(f"  {result.error}")
            return 0 if result.success else 1

        if args.command == "remove-model":
            result = asyncio.run(_remove(args, model))
            if args.json:
                _emit(result)
            else:
                print(
                    f"remove-model: {result.model} -> "
                    f"{'removed' if result.success else 'failed'} "
                    f"(verified_absent={result.verified_absent})"
                )
                if result.error:
                    print(f"  {result.error}")
            return 0 if result.success else 1

        if args.command == "model-fit":
            from .model_fit import model_fit_check

            result = model_fit_check(model, backend_hint=args.backend_hint)
            if args.json:
                _emit(result)
            else:
                print(
                    f"model-fit: {result.model} -> {result.fit} "
                    f"(suitable={result.suitable})"
                )
                print(f"  {result.reason}")
            return 0

        if args.command == "workload":
            from .workload import estimate_workload

            result = estimate_workload(model, context_tokens=args.context_tokens)
            if args.json:
                _emit(result)
            else:
                verdict = "fits" if result.fits else "does not fit"
                print(
                    f"workload: {result.model} -> {verdict} "
                    f"({result.estimated_memory_gb}GB estimated / "
                    f"{result.available_memory_gb}GB available)"
                )
            return 0

        if args.command == "capacity":
            from .capacity import model_capacity

            result = model_capacity(
                model,
                reserved_gb=args.reserved_gb,
                context_tokens=args.context_tokens,
            )
            if args.json:
                _emit(result)
            else:
                print(
                    f"capacity: {result.model} -> {result.max_instances} instance(s) "
                    f"({result.per_instance_gb}GB each)"
                )
            return 0
    except ImportError as exc:
        _emit(
            {
                "ok": False,
                "message": "deepiri-gpu-utils is required for hardware detection",
                "error": str(exc),
            }
        )
        return 2

    raise AssertionError(f"Unhandled management command: {args.command}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command in EXISTING_COMMANDS:
        return _run_existing(args)
    return _run_management(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a hyperparameter grid in separate Docker containers."""

import argparse
import copy
import itertools
import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")

    return cast(dict[str, Any], value)


def set_nested_value(
    config: dict[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    """Replace a nested configuration value using dotted notation."""
    keys = dotted_key.split(".")

    if not keys or any(not key for key in keys):
        raise ValueError(f"Invalid grid key: {dotted_key}")

    current = config

    for key in keys[:-1]:
        nested = current.get(key)

        if not isinstance(nested, dict):
            raise ValueError(f"Configuration section not found: {dotted_key}")

        current = cast(dict[str, Any], nested)

    if keys[-1] not in current:
        raise ValueError(f"Configuration parameter not found: {dotted_key}")

    current[keys[-1]] = value


def format_value(value: Any) -> str:
    """Create a filename-safe representation of a parameter value."""
    raw = format(value, ".10g") if isinstance(value, float) else str(value)

    return re.sub(r"[^A-Za-z0-9_-]", "p", raw)


def create_run_name(
    config: dict[str, Any],
    combination: dict[str, Any],
) -> str:
    """Create a descriptive, unique run name."""
    model_type = str(config["model"]["type"])
    seed = int(config["seed"])

    parts = [model_type]

    for key, value in combination.items():
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}{format_value(value)}")

    parts.append(f"seed{seed}")
    parts.append(uuid.uuid4().hex[:8])

    return "-".join(parts)


def get_git_commit() -> str:
    """Read the Git commit from the host repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def run_grid(
    config_path: Path,
    dry_run: bool = False,
) -> None:
    """Execute each grid combination in its own container."""
    grid_config = read_yaml(config_path)

    base_path = Path(str(grid_config["base"]))

    if not base_path.is_absolute() and not base_path.exists():
        base_path = config_path.parent / base_path

    base_config = read_yaml(base_path)

    raw_grid = grid_config["grid"]

    if not isinstance(raw_grid, dict) or not raw_grid:
        raise ValueError("Grid must be a non-empty mapping")

    grid = cast(dict[str, Any], raw_grid)

    keys = list(grid.keys())
    values: list[list[Any]] = []

    for key in keys:
        options = grid[key]

        if not isinstance(options, list) or not options:
            raise ValueError(f"Grid parameter {key} must have a non-empty list")

        values.append(options)

    combinations = list(itertools.product(*values))

    print(
        f"Base configuration: {base_path}",
        flush=True,
    )
    print(
        f"Total combinations: {len(combinations)}",
        flush=True,
    )

    for number, selected_values in enumerate(
        combinations,
        start=1,
    ):
        combination = dict(zip(keys, selected_values, strict=True))

        run_config = copy.deepcopy(base_config)

        for key, value in combination.items():
            set_nested_value(run_config, key, value)

        run_config["run_name"] = create_run_name(
            run_config,
            combination,
        )

        print(
            f"\n[{number}/{len(combinations)}] {run_config['run_name']}",
            flush=True,
        )
        print(
            "Parameters:",
            json.dumps(combination),
            flush=True,
        )

        if dry_run:
            continue

        with tempfile.TemporaryDirectory(prefix="hw3-grid-") as temporary:
            generated_config = Path(temporary) / "experiment.yaml"

            generated_config.write_text(
                yaml.safe_dump(
                    run_config,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            command = [
                "docker",
                "compose",
                "run",
                "--rm",
                "--env",
                f"GIT_COMMIT={get_git_commit()}",
                "--volume",
                (f"{generated_config.resolve()}:/tmp/hw3-experiment.yaml:ro"),
                "trainer",
                "python",
                "-m",
                "ds_infrastructure_hw.experiments.train",
                "--config",
                "/tmp/hw3-experiment.yaml",
            ]

            subprocess.run(command, check=True)

    print(
        "\nGrid completed." if not dry_run else "\nDry run completed. No training started.",
        flush=True,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    arguments = parser.parse_args()

    run_grid(
        config_path=arguments.config,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    main()

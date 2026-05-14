"""Run the SafeMetaDrive SAC-family training jobs in a fixed sequence."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    script: str
    use_lagrangian: bool


ALGORITHM_SPECS: Sequence[AlgorithmSpec] = (
    AlgorithmSpec("sac", "train_sacl.py", False),
    AlgorithmSpec("sacl", "train_sacl.py", True),
    AlgorithmSpec("sac_ra", "train_sacl_resact.py", False),
    AlgorithmSpec("sacl_ra", "train_sacl_resact.py", True),
)
ALGORITHM_ORDER = tuple(spec.name for spec in ALGORITHM_SPECS)


@dataclass(frozen=True)
class RunResult:
    name: str
    command: List[str]
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_algorithm_subset(value: str | None) -> List[str]:
    if value is None or not value.strip():
        return list(ALGORITHM_ORDER)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in requested if name not in ALGORITHM_ORDER]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown algorithm(s): {}. Expected one or more of {}".format(
                ", ".join(unknown),
                ", ".join(ALGORITHM_ORDER),
            )
        )
    requested_set = set(requested)
    return [name for name in ALGORITHM_ORDER if name in requested_set]


def build_command(
    spec: AlgorithmSpec,
    python_executable: str = sys.executable,
    traffic_density: float | None = None,
) -> List[str]:
    command = [
        python_executable,
        spec.script,
        "--task",
        "SafeMetaDrive",
        "--use_lagrangian",
        str(spec.use_lagrangian),
    ]
    if traffic_density is not None:
        command.extend(["--traffic_density", f"{float(traffic_density):g}"])
    return command


def selected_specs(algorithms: Iterable[str]) -> List[AlgorithmSpec]:
    selected = set(algorithms)
    return [spec for spec in ALGORITHM_SPECS if spec.name in selected]


def run_suite(
    algorithms: Iterable[str],
    *,
    dry_run: bool = False,
    traffic_density: float | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess] = subprocess.run,
    python_executable: str = sys.executable,
) -> List[RunResult]:
    results = []
    for spec in selected_specs(algorithms):
        command = build_command(spec, python_executable=python_executable, traffic_density=traffic_density)
        print("[{}] {}".format(spec.name, shlex.join(command)), flush=True)
        if dry_run:
            results.append(RunResult(spec.name, command, 0))
            continue
        completed = runner(command)
        results.append(RunResult(spec.name, command, int(completed.returncode)))
    return results


def print_summary(results: Sequence[RunResult]) -> None:
    print("\nSAC suite summary:")
    for result in results:
        status = "OK" if result.ok else "FAILED({})".format(result.returncode)
        print("- {}: {}".format(result.name, status))


def exit_code(results: Sequence[RunResult]) -> int:
    return 0 if all(result.ok for result in results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially train sac, sacl, sac_ra, and sacl_ra on SafeMetaDrive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the child training commands without launching them.",
    )
    parser.add_argument(
        "--algorithms",
        type=parse_algorithm_subset,
        default=list(ALGORITHM_ORDER),
        metavar="NAMES",
        help="Comma-separated subset from: {}".format(", ".join(ALGORITHM_ORDER)),
    )
    parser.add_argument(
        "--traffic-density",
        "--traffic_density",
        dest="traffic_density",
        type=float,
        default=None,
        help="Override traffic_density for every child training run. Omit to use each training script default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_suite(args.algorithms, dry_run=args.dry_run, traffic_density=args.traffic_density)
    print_summary(results)
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())

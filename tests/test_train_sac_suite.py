import argparse
import subprocess

import pytest

from train_sac_suite import (
    ALGORITHM_ORDER,
    ALGORITHM_SPECS,
    build_command,
    build_parser,
    exit_code,
    parse_algorithm_subset,
    run_suite,
)


def test_default_algorithm_order_is_fixed():
    assert ALGORITHM_ORDER == ("sac", "sacl", "sac_ra", "sacl_ra")
    assert parse_algorithm_subset(None) == list(ALGORITHM_ORDER)


def test_parse_algorithm_subset_uses_canonical_order():
    assert parse_algorithm_subset("sacl,sac") == ["sac", "sacl"]
    assert parse_algorithm_subset(" sac_ra , sacl_ra ") == ["sac_ra", "sacl_ra"]


def test_parse_algorithm_subset_rejects_unknown_names():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_algorithm_subset("sac,ppo")


def test_build_command_for_each_algorithm():
    results = run_suite(ALGORITHM_ORDER, dry_run=True, python_executable="python")
    assert [result.command for result in results] == [
        ["python", "train_sacl.py", "--task", "SafeMetaDrive", "--use_lagrangian", "False"],
        ["python", "train_sacl.py", "--task", "SafeMetaDrive", "--use_lagrangian", "True"],
        ["python", "train_sacl_resact.py", "--task", "SafeMetaDrive", "--use_lagrangian", "False"],
        ["python", "train_sacl_resact.py", "--task", "SafeMetaDrive", "--use_lagrangian", "True"],
    ]


def test_build_command_for_single_spec():
    result = run_suite(["sac_ra"], dry_run=True, python_executable="python")[0]
    assert result.command == build_command(
        next(spec for spec in ALGORITHM_SPECS if spec.name == "sac_ra"),
        python_executable="python",
    )


def test_traffic_density_is_forwarded_when_configured():
    results = run_suite(
        ["sac", "sacl_ra"],
        dry_run=True,
        python_executable="python",
        traffic_density=0.15,
    )
    assert [result.command for result in results] == [
        [
            "python",
            "train_sacl.py",
            "--task",
            "SafeMetaDrive",
            "--use_lagrangian",
            "False",
            "--traffic_density",
            "0.15",
        ],
        [
            "python",
            "train_sacl_resact.py",
            "--task",
            "SafeMetaDrive",
            "--use_lagrangian",
            "True",
            "--traffic_density",
            "0.15",
        ],
    ]


def test_parser_accepts_hyphenated_and_underscored_traffic_density():
    assert build_parser().parse_args(["--traffic-density", "0.2"]).traffic_density == 0.2
    assert build_parser().parse_args(["--traffic_density", "0.3"]).traffic_density == 0.3


def test_failure_does_not_stop_later_algorithms():
    calls = []
    returncodes = iter([0, 2, 0, 3])

    def runner(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, next(returncodes))

    results = run_suite(ALGORITHM_ORDER, runner=runner, python_executable="python")

    assert [result.name for result in results] == list(ALGORITHM_ORDER)
    assert [result.returncode for result in results] == [0, 2, 0, 3]
    assert len(calls) == 4
    assert exit_code(results) == 1


def test_success_exit_code_is_zero():
    results = run_suite(["sac"], dry_run=True, python_executable="python")
    assert exit_code(results) == 0

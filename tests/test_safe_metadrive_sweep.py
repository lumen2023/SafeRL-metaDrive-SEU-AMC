from safe_metadrive_sweep import (
    MIXED_DEFAULT_SCENE,
    SAFE_METADRIVE_PROJECT,
    SAFE_METADRIVE_SWEEP_SCENES,
    append_scene_tag_to_run_name,
    build_safe_metadrive_child_command,
    is_mixed_default_scene,
    normalize_safe_metadrive_scene,
    safe_metadrive_group,
    strip_cli_options,
)


def test_safe_metadrive_scene_sequence_is_fixed():
    assert SAFE_METADRIVE_SWEEP_SCENES == (
        "mixed_default",
        "straight_4lane",
        "ramp_merge",
        "t_intersection",
        "roundabout",
        "lane_change_bottleneck",
    )


def test_safe_metadrive_project_and_group_naming():
    assert SAFE_METADRIVE_PROJECT == "metadrive-fast-rl"
    assert safe_metadrive_group(MIXED_DEFAULT_SCENE) == "scene-mixed_default"
    assert safe_metadrive_group("ramp_merge") == "scene-ramp_merge"


def test_normalize_safe_metadrive_scene_rejects_unknown_values():
    assert normalize_safe_metadrive_scene("roundabout") == "roundabout"
    assert is_mixed_default_scene("mixed_default") is True
    assert is_mixed_default_scene("straight_4lane") is False


def test_strip_cli_options_removes_scene_sweep_pairs_and_equals_forms():
    argv = [
        "--task",
        "SafeMetaDrive",
        "--safe_metadrive_sweep",
        "True",
        "--epoch",
        "5",
        "--safe_metadrive_scene=ramp_merge",
        "--name",
        "demo",
    ]
    assert strip_cli_options(argv, ("--safe_metadrive_sweep", "--safe_metadrive_scene")) == [
        "--task",
        "SafeMetaDrive",
        "--epoch",
        "5",
        "--name",
        "demo",
    ]


def test_build_safe_metadrive_child_command_forces_single_scene_run():
    parent_argv = [
        "--task",
        "SafeMetaDrive",
        "--epoch",
        "2",
        "--safe_metadrive_sweep",
        "True",
        "--safe_metadrive_scene",
        "mixed_default",
    ]
    command = build_safe_metadrive_child_command("train_ppol.py", parent_argv, "t_intersection")

    assert command[1] == "train_ppol.py"
    assert "--safe_metadrive_sweep" in command
    assert "--safe_metadrive_scene" in command
    assert command[-1] == "t_intersection"
    assert "False" in command
    assert command.count("--safe_metadrive_sweep") == 1
    assert command.count("--safe_metadrive_scene") == 1


def test_append_scene_tag_to_run_name_is_idempotent():
    tagged = append_scene_tag_to_run_name("PPOL-demo", "straight_4lane")
    assert tagged == "PPOL-demo_scene-straight_4lane"
    assert append_scene_tag_to_run_name(tagged, "straight_4lane") == tagged

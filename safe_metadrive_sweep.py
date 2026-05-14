import sys
from typing import Iterable, List, Optional


SAFE_METADRIVE_PROJECT = "metadrive-rl"
MIXED_DEFAULT_SCENE = "mixed_default"
SAFE_METADRIVE_SWEEP_SCENES = (
    MIXED_DEFAULT_SCENE,
    "straight_4lane",
    "ramp_merge",
    "t_intersection",
    "roundabout",
    "lane_change_bottleneck",
)


def normalize_safe_metadrive_scene(scene: str) -> str:
    scene = str(scene or MIXED_DEFAULT_SCENE)
    if scene not in SAFE_METADRIVE_SWEEP_SCENES:
        raise ValueError(
            "Unsupported SafeMetaDrive scene {!r}. Expected one of {}".format(
                scene, SAFE_METADRIVE_SWEEP_SCENES
            )
        )
    return scene


def is_mixed_default_scene(scene: str) -> bool:
    return normalize_safe_metadrive_scene(scene) == MIXED_DEFAULT_SCENE


def safe_metadrive_group(scene: str) -> Optional[str]:
    scene = normalize_safe_metadrive_scene(scene)
    if is_mixed_default_scene(scene):
        return None
    return scene


def append_scene_tag_to_run_name(name: str, scene: str) -> str:
    scene_tag = safe_metadrive_group(scene)
    if not scene_tag:
        return str(name)
    if scene_tag in str(name):
        return str(name)
    return "{}_{}".format(name, scene_tag)


def _normalize_option_names(option_names: Iterable[str]) -> List[str]:
    normalized = []
    for option_name in option_names:
        option_name = str(option_name)
        normalized.append(option_name)
        if "_" in option_name:
            normalized.append(option_name.replace("_", "-"))
        if "-" in option_name:
            normalized.append(option_name.replace("-", "_"))
    deduped = []
    seen = set()
    for option_name in normalized:
        if option_name in seen:
            continue
        seen.add(option_name)
        deduped.append(option_name)
    return deduped


def strip_cli_options(argv: Iterable[str], option_names: Iterable[str]) -> List[str]:
    option_names = set(_normalize_option_names(option_names))
    argv = list(argv)
    stripped = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        matched_option = None
        for option_name in option_names:
            if token == option_name:
                matched_option = option_name
                break
            if token.startswith(option_name + "="):
                matched_option = option_name
                break
        if matched_option is None:
            stripped.append(token)
            continue
        if token == matched_option:
            skip_next = True
    return stripped


def build_safe_metadrive_child_command(script_path: str, parent_argv: Iterable[str], scene: str) -> List[str]:
    scene = normalize_safe_metadrive_scene(scene)
    child_argv = strip_cli_options(parent_argv, ("--safe_metadrive_sweep", "--safe_metadrive_scene"))
    return [
        sys.executable,
        script_path,
        *child_argv,
        "--safe_metadrive_sweep",
        "False",
        "--safe_metadrive_scene",
        scene,
    ]

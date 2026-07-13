from __future__ import annotations

from maniskill_exp import load_task_registry

DEFAULT_CAMERA_NAME = "left_camera"
DEFAULT_CAMERA_NAMES = ("left_camera", "right_camera")
DEFAULT_SIDE_VIEW_Y_OFFSET = 0.45

CAMERA_LOOK_ATS = {camera_name: {} for camera_name in DEFAULT_CAMERA_NAMES}

TASK_CAMERA_ANCHORS = {
    "LiftPegUpright-v1": {"eye": [0.3, 0.0, 0.6], "target": [-0.1, 0.0, 0.1]},
    "PegInsertionSide-v1": {"eye": [0.0, -0.3, 0.2], "target": [0.0, 0.0, 0.1]},
    "PickCube-v1": {"eye": [0.3, 0.0, 0.6], "target": [-0.1, 0.0, 0.1]},
    "PlugCharger-v1": {"eye": [0.3, 0.0, 0.6], "target": [-0.1, 0.0, 0.1]},
    "PokeCube-v1": {"eye": [0.3, 0.0, 0.6], "target": [-0.1, 0.0, 0.1]},
    "PullCube-v1": {"eye": [-0.5, 0.0, 0.25], "target": [0.2, 0.0, -0.5]},
    "PullCubeTool-v1": {"eye": [0.3, 0.0, 0.5], "target": [-0.1, 0.0, 0.1]},
    "PushCube-v1": {"eye": [0.3, 0.0, 0.6], "target": [-0.1, 0.0, 0.1]},
    "PushT-v1": {"eye": [0.3, 0.0, 0.6], "target": [-0.1, 0.0, 0.1]},
    "RollBall-v1": {"eye": [-0.1, 0.9, 0.3], "target": [0.0, 0.0, 0.0]},
    "StackCube-v1": {"eye": [0.3, 0.0, 0.6], "target": [-0.1, 0.0, 0.1]},
}


def validate_camera_name(camera_name: str) -> str:
    if camera_name not in CAMERA_LOOK_ATS:
        supported = ", ".join(sorted(CAMERA_LOOK_ATS))
        raise KeyError(f"Unknown camera preset: {camera_name}. Expected one of: {supported}")
    return camera_name


def resolve_camera_look_at(task_name: str, camera_name: str) -> dict[str, list[float]]:
    validate_camera_name(camera_name)
    if task_name not in TASK_CAMERA_ANCHORS:
        raise KeyError(f"Missing default ManiSkill base-camera anchor for task: {task_name}")

    anchor = TASK_CAMERA_ANCHORS[task_name]
    eye = [float(value) for value in anchor["eye"]]
    target = [float(value) for value in anchor["target"]]

    if abs(eye[1] - target[1]) < 1e-6:
        side_offset = DEFAULT_SIDE_VIEW_Y_OFFSET
        eye[1] = target[1] + (side_offset if camera_name == "left_camera" else -side_offset)
    elif camera_name == "right_camera":
        eye[1] = 2.0 * target[1] - eye[1]

    return {"eye": eye, "target": target}


def validate_task_camera(task_name: str, camera_name: str, *, task_registry: dict[str, dict] | None = None) -> str:
    validate_camera_name(camera_name)
    registry = task_registry or load_task_registry()
    if task_name not in registry:
        raise KeyError(f"Unknown ManiSkill task: {task_name}")
    allowed_cameras = tuple(registry[task_name].get("camera_names", DEFAULT_CAMERA_NAMES))
    if camera_name not in allowed_cameras:
        allowed = ", ".join(allowed_cameras)
        raise ValueError(
            f"Camera {camera_name} is not configured for task {task_name}. "
            f"Use one of the expert-view cameras: {allowed}"
        )
    return camera_name

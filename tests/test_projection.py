import math

import numpy as np

from dlp.data.nusc_projection import box_corners, global_to_camera, project_box, project_point, quat_to_rot
from make_synthetic import cam_rotation, yaw_quat

K = [[1000.0, 0, 800.0], [0, 1000.0, 450.0], [0, 0, 1]]
IDENT_POSE = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
FRONT = {"translation": [0, 0, 0], "rotation": cam_rotation(0), "camera_intrinsic": K}


def test_quaternion_yaw():
    R = quat_to_rot(yaw_quat(math.pi / 2))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)


def test_point_straight_ahead_projects_to_principal_point():
    uv = project_point([10, 0, 0], IDENT_POSE, FRONT)
    assert np.allclose(uv, [800, 450], atol=1e-6)


def test_left_is_left_and_up_is_up():
    u_left, _ = project_point([10, 2, 0], IDENT_POSE, FRONT)   # ego +y = left
    _, v_up = project_point([10, 0, 1], IDENT_POSE, FRONT)     # ego +z = up
    assert u_left < 800 and v_up < 450
    assert np.isclose(u_left, 800 - 1000 * 2 / 10)


def test_behind_camera_is_not_visible():
    assert project_point([-5, 0, 0], IDENT_POSE, FRONT) is None
    corners = box_corners([-5, 0, 0], [2, 4, 1.5], [1, 0, 0, 0])
    assert project_box(global_to_camera(corners, IDENT_POSE, FRONT), K) is None


def test_box_size_convention_and_ego_pose():
    # nuScenes size = [w, l, h]; length runs along the box x axis
    c = box_corners([0, 0, 0], [2.0, 4.0, 1.0], [1, 0, 0, 0])
    assert np.isclose(c[0].max(), 2.0) and np.isclose(c[1].max(), 1.0)
    # ego at (100, 0) facing +y: a point at (100, 10) is straight ahead
    pose = {"translation": [100, 0, 0], "rotation": yaw_quat(math.pi / 2)}
    assert np.allclose(project_point([100, 10, 0], pose, FRONT), [800, 450], atol=1e-6)


def test_back_camera_sees_behind():
    back = {**FRONT, "rotation": cam_rotation(180)}
    assert np.allclose(project_point([-10, 0, 0], IDENT_POSE, back), [800, 450], atol=1e-6)


def test_truncated_box_is_clipped():
    corners = box_corners([3, 0, 0], [2, 8, 1.5], [1, 0, 0, 0])  # straddles the image plane
    box = project_box(global_to_camera(corners, IDENT_POSE, FRONT), K)
    assert box is not None and 0 <= box[0] and box[2] <= 1600 and box[3] <= 900

"""Generate a SYNTHETIC mini dataset in the exact DriveLM + nuScenes JSON layouts.

Purpose: run and test the whole pipeline (build -> predict -> eval -> tables ->
failure images) on a machine without the real data. The scenes are boxes on a
grey road rendered with nuScenes-like camera intrinsics/extrinsics; questions
follow the DriveLM perception wording we expect. Nothing produced from this data
is a research result.

    python scripts/make_synthetic.py --out data/synthetic --scenes 16 --samples 6
"""
import argparse
import json
import math
import os
import random
import uuid

import numpy as np
from PIL import Image, ImageDraw

from _common import rel
from dlp.data.nusc_projection import box_corners, global_to_camera, project_box, project_point
from dlp.data.templates import CAMERAS

CAM_YAW = {"CAM_FRONT": 0, "CAM_FRONT_LEFT": 55, "CAM_FRONT_RIGHT": -55,
           "CAM_BACK_LEFT": 110, "CAM_BACK": 180, "CAM_BACK_RIGHT": -110}
K = [[1266.4, 0.0, 800.0], [0.0, 1266.4, 450.0], [0.0, 0.0, 1.0]]
CATS = [  # name, size wlh, colour, description words, weight
    ("vehicle.car", (1.9, 4.6, 1.6), (200, 200, 205), "sedan", 0.45),
    ("vehicle.truck", (2.5, 7.5, 3.0), (170, 120, 60), "truck", 0.12),
    ("vehicle.bus.rigid", (2.9, 11.0, 3.4), (230, 190, 40), "bus", 0.06),
    ("human.pedestrian.adult", (0.7, 0.7, 1.75), (220, 80, 80), "pedestrian", 0.2),
    ("vehicle.bicycle", (0.6, 1.8, 1.4), (80, 200, 120), "cyclist", 0.07),
    ("movable_object.trafficcone", (0.4, 0.4, 0.8), (255, 140, 0), "traffic cone", 0.1),
]
GROUP_WORD = {"vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus.rigid": "bus",
              "human.pedestrian.adult": "pedestrian", "vehicle.bicycle": "bicycle",
              "movable_object.trafficcone": "traffic cone"}
PLURAL = {"car": "cars", "truck": "trucks", "bus": "buses", "pedestrian": "pedestrians", "bicycle": "bicycles",
          "traffic cone": "traffic cones"}
COLOURS = ["white", "black", "silver", "red", "blue"]


def tok():
    return uuid.uuid4().hex


def rot_to_quat(R):
    R = np.asarray(R)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    return [0.0, x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)]


def yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def cam_rotation(yaw_deg):
    y = math.radians(yaw_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # columns: cam x, y, z in ego frame
    return rot_to_quat(Rz @ base)


def where(cam):
    return {"CAM_FRONT": "front", "CAM_BACK": "back", "CAM_FRONT_LEFT": "front left",
            "CAM_FRONT_RIGHT": "front right", "CAM_BACK_LEFT": "back left", "CAM_BACK_RIGHT": "back right"}[cam]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--scenes", type=int, default=16)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--objects", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_scale", type=float, default=0.5, help="saved image size relative to 1600x900")
    args = p.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    uuid.uuid4 = lambda: uuid.UUID(int=rng.getrandbits(128))  # deterministic tokens

    out = rel(args.out)
    nroot = os.path.join(out, "nuscenes")
    meta = os.path.join(nroot, "v1.0-mini")
    os.makedirs(meta, exist_ok=True)
    for cam in CAMERAS:
        os.makedirs(os.path.join(nroot, "samples", cam), exist_ok=True)
    os.makedirs(os.path.join(out, "drivelm"), exist_ok=True)

    T = {k: [] for k in ["log", "scene", "sample", "sample_data", "calibrated_sensor", "sensor",
                         "ego_pose", "sample_annotation", "instance", "category", "attribute"]}
    cat_tok = {}
    for name, *_ in CATS:
        cat_tok[name] = tok()
        T["category"].append({"token": cat_tok[name], "name": name, "description": ""})
    attr_tok = {}
    for a in ["vehicle.moving", "vehicle.parked", "vehicle.stopped", "pedestrian.moving",
              "pedestrian.standing", "cycle.with_rider"]:
        attr_tok[a] = tok()
        T["attribute"].append({"token": attr_tok[a], "name": a, "description": ""})
    calib = {}
    for cam in CAMERAS:
        st = tok()
        T["sensor"].append({"token": st, "channel": cam, "modality": "camera"})
        calib[cam] = {"token": tok(), "sensor_token": st, "translation": [1.5, 0.0, 1.5],
                      "rotation": cam_rotation(CAM_YAW[cam]), "camera_intrinsic": K}
        T["calibrated_sensor"].append(calib[cam])

    drivelm = {}
    W, H = int(1600 * args.img_scale), int(900 * args.img_scale)
    for s in range(args.scenes):
        city = "boston-seaport" if s % 2 == 0 else "singapore-onenorth"
        cond = ["Day, clear", "Night, wet road", "Rain, heavy traffic", "Day, intersection"][s % 4]
        log_t, scene_t = tok(), tok()
        T["log"].append({"token": log_t, "location": city, "logfile": f"synthetic-{s}"})
        ego_yaw = rng.uniform(-math.pi, math.pi)
        ego_xy = np.array([rng.uniform(0, 500), rng.uniform(0, 500)])
        ego_v = rng.choice([0.0, 4.0, 8.0])
        fwd = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
        left = np.array([-fwd[1], fwd[0]])
        objs = []
        for _ in range(args.objects):
            name, size, colour, word, _w = rng.choices(CATS, weights=[c[4] for c in CATS])[0]
            rel_xy = fwd * rng.uniform(-35, 45) + left * rng.uniform(-14, 14)
            if np.linalg.norm(rel_xy) < 5:
                rel_xy += fwd * 8
            static = name.startswith("movable") or rng.random() < 0.4
            heading = ego_yaw + (rng.choice([0, math.pi]) if name.startswith("vehicle") else rng.uniform(-3, 3))
            speed = 0.0 if static else (1.3 if "pedestrian" in name else rng.uniform(3, 9))
            turn = 0.0 if static or rng.random() < 0.8 else rng.choice([-0.25, 0.25])
            inst = tok()
            T["instance"].append({"token": inst, "category_token": cat_tok[name]})
            objs.append({"name": name, "size": size, "colour": colour, "word": word, "inst": inst,
                         "xy": ego_xy + rel_xy, "heading": heading, "speed": speed, "turn": turn,
                         "paint": rng.choice(COLOURS), "prev_ann": None})
        sample_toks = [tok() for _ in range(args.samples)]
        T["scene"].append({"token": scene_t, "log_token": log_t, "name": f"scene-{s:04d}", "description": cond,
                           "first_sample_token": sample_toks[0], "last_sample_token": sample_toks[-1],
                           "nbr_samples": args.samples})
        key_frames = {}
        for k, st in enumerate(sample_toks):
            ts = 1_500_000_000_000_000 + s * 100_000_000 + k * 500_000
            T["sample"].append({"token": st, "scene_token": scene_t, "timestamp": ts,
                                "prev": sample_toks[k - 1] if k else "", "next": sample_toks[k + 1] if k + 1 < len(sample_toks) else ""})
            pose = {"token": tok(), "timestamp": ts, "translation": [float(ego_xy[0]), float(ego_xy[1]), 0.0],
                    "rotation": yaw_quat(ego_yaw)}
            T["ego_pose"].append(pose)
            anns = []
            for o in objs:
                if o["speed"] > 0:
                    attr = "pedestrian.moving" if "pedestrian" in o["name"] else (
                        "cycle.with_rider" if "bicycle" in o["name"] else "vehicle.moving")
                else:
                    attr = "pedestrian.standing" if "pedestrian" in o["name"] else (
                        "" if o["name"].startswith("movable") else rng.choice(["vehicle.parked", "vehicle.stopped"]))
                a = {"token": tok(), "sample_token": st, "instance_token": o["inst"],
                     "translation": [float(o["xy"][0]), float(o["xy"][1]), o["size"][2] / 2],
                     "size": list(o["size"]), "rotation": yaw_quat(o["heading"]),
                     "attribute_tokens": [attr_tok[attr]] if attr else [], "visibility_token": "4",
                     "prev": o["prev_ann"]["token"] if o["prev_ann"] else "", "next": ""}
                if o["prev_ann"]:
                    o["prev_ann"]["next"] = a["token"]
                o["prev_ann"] = a
                T["sample_annotation"].append(a)
                anns.append((o, a))
            # render cameras + collect visible boxes
            vis = {cam: [] for cam in CAMERAS}
            paths = {}
            for cam in CAMERAS:
                fname = f"samples/{cam}/syn_{s:03d}_{k:02d}__{cam}.jpg"
                T["sample_data"].append({"token": tok(), "sample_token": st, "ego_pose_token": pose["token"],
                                         "calibrated_sensor_token": calib[cam]["token"], "filename": fname,
                                         "fileformat": "jpg", "is_key_frame": True, "width": 1600, "height": 900,
                                         "timestamp": ts})
                paths[cam] = "../nuscenes/" + fname
                night = "Night" in cond
                img = Image.new("RGB", (W, H), (35, 35, 50) if night else (150, 170, 190))
                d = ImageDraw.Draw(img)
                d.rectangle([0, H * 0.5, W, H], fill=(40, 40, 40) if night else (95, 95, 95))
                items = []
                for o, a in anns:
                    cc = global_to_camera(box_corners(a["translation"], a["size"], a["rotation"]), pose, calib[cam])
                    box = project_box(cc, K)
                    if box is None or min(box[2] - box[0], box[3] - box[1]) < 6:
                        continue
                    depth = float(np.mean(cc[2]))
                    centre = project_point(a["translation"], pose, calib[cam])
                    items.append((depth, o, a, box, centre))
                for depth, o, a, box, centre in sorted(items, key=lambda t: -t[0]):
                    sc = args.img_scale
                    col = tuple(int(c * (0.6 if night else 1.0)) for c in o["colour"])
                    d.rectangle([box[0] * sc, box[1] * sc, box[2] * sc, box[3] * sc], fill=col, outline=(20, 20, 20))
                    vis[cam].append((depth, o, a, box, centre))
                img.save(os.path.join(nroot, fname), quality=80)
            # move the world for the next keyframe (0.5 s)
            for o in objs:
                o["heading"] += o["turn"]
                o["xy"] = o["xy"] + 0.5 * o["speed"] * np.array([math.cos(o["heading"]), math.sin(o["heading"])])
            ego_xy = ego_xy + 0.5 * ego_v * fwd

            # ---- DriveLM annotations for this keyframe
            cands = []
            for cam in CAMERAS:
                for depth, o, a, box, centre in vis[cam]:
                    if depth < 45 and not o["name"].startswith("movable"):
                        x, y = centre if centre and 0 <= centre[0] <= 1600 and 0 <= centre[1] <= 900 else \
                            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        cands.append((depth, cam, x, y, o, box))
            cands.sort(key=lambda c: c[0])
            chosen, seen = [], set()
            for c in cands:
                if c[4]["inst"] not in seen:
                    seen.add(c[4]["inst"])
                    chosen.append(c)
                if len(chosen) == 4:
                    break
            infos, qa, descs, tag_strs = {}, [], [], []
            for i, (depth, cam, x, y, o, box) in enumerate(chosen, 1):
                tag = f"<c{i},{cam},{x:.1f},{y:.1f}>"
                desc = f"{o['paint'].capitalize()} {o['word']}." if o["word"] in ("sedan", "truck", "bus") else \
                    f"{o['word'].capitalize()} in {o['paint']}."
                moving = o["speed"] > 0
                if rng.random() < 0.08:  # simulated annotator disagreement, caught by the label check
                    moving = not moving
                status = ("Turn left." if o["turn"] > 0 else "Turn right." if o["turn"] < 0 else "Going ahead.") \
                    if moving else "Stopped."
                cat_word = "Pedestrian" if "pedestrian" in o["name"] else "Vehicle"
                infos[tag] = {"Category": cat_word, "Status": "Moving" if moving else "Stationary",
                              "Visual_description": desc, "2d_bbox": [round(v, 1) for v in box]}
                descs.append(f"a {desc.rstrip('.').lower()} to the {where(cam)} of the ego vehicle")
                tag_strs.append(tag)
                obs = ("Standing." if "pedestrian" in o["name"] else "Parked.") if not moving else "Moving."
                qa.append({"Q": f"What is the moving status of object {tag}?", "A": status})
                qa.append({"Q": f"What is the observed status of object {tag}?", "A": obs})
                qa.append({"Q": f"What is the visual description of {tag}?", "A": desc})
            if chosen:
                qa.insert(0, {"Q": "What are the important objects in the current scene? Those objects will be "
                                   "considered for the future reasoning and driving decision.",
                              "A": "There is " + ", ".join(descs) + ". The IDs of these objects are " +
                                   ", ".join(tag_strs) + "."})
                qa.append({"Q": "What is the relative positioning of the important objects in the current scene?",
                           "A": f"{tag_strs[0]} is the closest object."})
            # region-level questions, as in DriveLM v1.1 (region = camera)
            for cam in CAMERAS:
                near = [(o, a) for depth, o, a, box, centre in vis[cam] if depth < 45]
                region = where(cam)
                groups = {}
                for o, a in near:
                    groups.setdefault(GROUP_WORD[o["name"]], []).append(o)
                if groups:
                    parts = [f"{'one' if len(v) == 1 else 'two' if len(v) == 2 else 'many'} "
                             f"{g if len(v) == 1 else PLURAL[g]}" for g, v in sorted(groups.items())]
                    qa.append({"Q": f"What are objects to the {region} of the ego car?",
                               "A": f"There {'is' if len(near) == 1 else 'are'} {' and '.join(parts)} to the {region} of the ego car."})
                    g, v = max(groups.items(), key=lambda kv: len(kv[1]))
                    if g not in ("traffic cone",):
                        verb = "moving" if v[0]["speed"] > 0 else ("standing" if g == "pedestrian" else "parked")
                        if len(v) == 1:
                            qa.append({"Q": f"What is the status of the {g} that is to the {region} of the ego car?",
                                       "A": f"One {g} is {verb}."})
                        else:
                            qa.append({"Q": f"What is the status of the {PLURAL[g]} that are to the {region} of the ego car?",
                                       "A": f"Many {PLURAL[g]} are {verb}."})
                parked_cars = any(o["name"] == "vehicle.car" and o["speed"] == 0 for o, a in near)
                moving_peds = any("pedestrian" in o["name"] and o["speed"] > 0 for o, a in near)
                qa.append({"Q": f"Are there parked cars to the {region} of the ego car?", "A": "Yes." if parked_cars else "No."})
                qa.append({"Q": f"Are there moving pedestrians to the {region} of the ego car?", "A": "Yes." if moving_peds else "No."})
            qa.append({"Q": "Is there any traffic element in the front view?", "A": "No."})
            qa.append({"Q": "Please describe the current scene.", "A": "There are several vehicles around the ego car."})
            key_frames[st] = {"key_object_infos": infos, "QA": {"perception": qa, "prediction": [], "planning": [],
                                                                "behavior": []}, "image_paths": paths}
        drivelm[scene_t] = {"scene_description": cond, "key_frames": key_frames}

    # one DriveLM frame whose token is not in the metadata (exercises that label check)
    first_scene = next(iter(drivelm.values()))
    first_scene["key_frames"]["deadbeef" * 4] = {"key_object_infos": {}, "QA": {"perception": []}, "image_paths": {}}

    for name, rows in T.items():
        json.dump(rows, open(os.path.join(meta, f"{name}.json"), "w"))
    json.dump(drivelm, open(os.path.join(out, "drivelm", "v1_1_train_nus.json"), "w"), indent=1)
    n_qa = sum(len(f["QA"]["perception"]) for sc in drivelm.values() for f in sc["key_frames"].values())
    print(f"synthetic: {args.scenes} scenes, {len(T['sample'])} keyframes, {len(T['sample_annotation'])} annotations, "
          f"{n_qa} perception QA -> {out}")


if __name__ == "__main__":
    main()

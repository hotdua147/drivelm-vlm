import json
import os
import subprocess
import sys

from dlp.eval.metrics import balanced_accuracy, evaluate, macro_f1
from dlp.utils.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_macro_f1_and_balanced_accuracy():
    assert macro_f1(["a", "a", "b"], ["a", "a", "b"]) == 100.0
    assert macro_f1(["a", "a", "b", "b"], ["a", "a", "a", "a"]) == round((66.67 + 0) / 2, 2)
    assert balanced_accuracy(["y", "y", "y", "n"], ["y", "y", "y", "y"]) == 50.0


def test_evaluate_small(tmp_path):
    gt = tmp_path / "gt"
    gt.mkdir()
    json.dump({"CAM_FRONT": [{"box2d": [90, 90, 110, 110]}]}, open(gt / "s1.json", "w"))
    preds = [
        {"qtype": "P1", "sample_token": "s1", "question": "", "answer": "", "answer_label": None,
         "tags": [{"in": "answer", "cam": "CAM_FRONT", "x": 100, "y": 100}],
         "prediction": "<c1,CAM_FRONT,104,100> <c2,CAM_FRONT,900,800>", "city": "boston", "condition": "day"},
        {"qtype": "P2", "sample_token": "s1", "question": "Options: A. Going ahead. B. Stopped.", "answer": "B.",
         "answer_label": "stopped", "tags": [], "prediction": "B."},
    ]
    preds += [
        {"qtype": "P5", "sample_token": "s1", "question": "What are objects to the front of the ego car?",
         "answer": "", "answer_label": "barrier|car", "tags": [], "prediction": "There are two cars and a bus."},
        {"qtype": "P6", "sample_token": "s1", "question": "", "answer": "", "answer_label": "stopped", "tags": [],
         "prediction": "Many trucks are parked."},
    ]
    r = evaluate(preds, str(gt))
    assert r["P5"] == {"precision": 50.0, "recall": 50.0, "f1": 50.0}
    assert r["P6"]["accuracy"] == 100.0
    assert r["P1"]["tau50"] == {"precision": 50.0, "recall": 100.0, "f1": 66.67}
    assert r["grounding_rate"] == 50.0 and r["hallucination_rate"] == 50.0
    assert r["P2"]["accuracy"] == 100.0
    assert r["subsets"]["city=boston"]["n"] == 1


def test_config_inheritance():
    cfg = load_config(os.path.join(ROOT, "configs/exp/r1_6cam.yaml"))
    assert cfg["model"]["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert cfg["train"]["enabled"] and cfg["train"]["camera_dropout"] == 0.15 and cfg["train"]["grad_accum"] == 16
    abl = load_config(os.path.join(ROOT, "configs/exp/abl_front_only.yaml"))
    assert abl["ablation"]["cameras"] == ["CAM_FRONT"] and not abl["train"]["enabled"]


def _run(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def test_end_to_end_on_synthetic(tmp_path):
    """make_synthetic -> build -> predict (prior, mock) -> eval -> tables -> failure images."""
    out = tmp_path / "syn"
    _run("scripts/make_synthetic.py", "--out", str(out), "--scenes", "8", "--samples", "3")
    sets = [f"paths.drivelm_json={out}/drivelm/v1_1_train_nus.json", f"paths.nusc_root={out}/nuscenes",
            f"paths.image_root={out}/nuscenes", f"paths.processed={out}/processed", f"paths.runs_dir={tmp_path}/runs",
            "data.val_frac=0.4", "data.thresholds.min_sample_found_rate=0.9"]
    log = _run("scripts/build_perception.py", "--config", "configs/exp/smoke_mock.yaml", "--set", *sets)
    assert "LABEL CHECK PASSED" in log
    lc = json.load(open(out / "processed" / "label_check.json"))
    assert lc["rates"]["tag_link_rate"] == 1.0 and lc["rates"]["category_agreement"] == 1.0
    # the generator answers "parked cars" from speed == 0 (parked or stopped): the label check must find that definition
    assert lc["p4_best_variant"] == "parked_incl_stopped"
    assert lc["status_pairs_top"] and lc["p4_by_template"]
    rows = [json.loads(l) for l in open(out / "processed" / "train.jsonl")]
    assert any("nusc_disagree" in r["label_flags"] for r in rows if r["qtype"] == "P2o")  # injected status flips
    runs = {}
    for exp in ("smoke_prior", "smoke_mock"):
        run = _run("scripts/predict.py", "--config", f"configs/exp/{exp}.yaml", "--set", *sets).strip().splitlines()[-1]
        _run("scripts/eval_perception.py", "--run", run)
        runs[exp] = json.load(open(os.path.join(run, "_report.json")))
    assert runs["smoke_mock"]["P1"]["tau100"]["f1"] > 50       # noisy oracle finds most objects
    assert runs["smoke_prior"]["warnings"]                       # constant prior must trigger COLLAPSE
    assert not any("P2" in w for w in runs["smoke_mock"]["warnings"])
    tables = _run("scripts/make_tables.py", "--runs_dir", str(tmp_path / "runs"), "--out", str(tmp_path / "t.md"))
    assert "smoke_mock" in tables and "Table 3" in tables and "Table 4" in tables
    mock_run = [d for d in os.listdir(tmp_path / "runs") if "smoke_mock" in d][0]
    _run("scripts/viz_failures.py", "--run", str(tmp_path / "runs" / mock_run), "--n", "2")
    assert len(os.listdir(tmp_path / "runs" / mock_run / "failures")) == 3  # 2 images + index.json


def test_label_check_blocks_bad_data(tmp_path):
    out = tmp_path / "syn"
    _run("scripts/make_synthetic.py", "--out", str(out), "--scenes", "4", "--samples", "2")
    sets = [f"paths.drivelm_json={out}/drivelm/v1_1_train_nus.json", f"paths.nusc_root={out}/nuscenes",
            f"paths.image_root={out}/nuscenes", f"paths.processed={out}/processed",
            "data.thresholds.min_status_agreement=0.999"]
    proc = subprocess.run([sys.executable, "scripts/build_perception.py", "--config", "configs/exp/smoke_mock.yaml",
                           "--set", *sets], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 1 and "LABEL CHECK FAILED" in proc.stderr


def test_p4_answer_verified_by_nuscenes_boxes():
    from dlp.data.build import p4_from_nuscenes
    gt = {"CAM_FRONT_LEFT": [{"category": "vehicle.car", "group": "car", "attribute": "vehicle.stopped"},
                             {"category": "vehicle.trailer", "group": "truck", "attribute": "vehicle.parked"}],
          "CAM_FRONT": [{"category": "vehicle.car", "group": "car", "attribute": "vehicle.parked"}],
          "CAM_BACK": [{"category": "human.pedestrian.adult", "group": "pedestrian", "attribute": "pedestrian.standing"}]}
    q = "Are there parked cars to the front left of the ego car?"
    assert p4_from_nuscenes(q, gt) == "no"                                   # stopped is not parked
    assert p4_from_nuscenes(q, gt, parked_includes_stopped=True) == "yes"
    assert p4_from_nuscenes(q, gt, neighbours=True) == "yes"                 # parked car in CAM_FRONT
    assert p4_from_nuscenes("Are there parked trucks to the front left of the ego car?", gt) == "no"  # trailer != truck
    assert p4_from_nuscenes("Are there parked trailers to the front left of the ego car?", gt) == "yes"
    assert p4_from_nuscenes("Are there standing pedestrians to the back of the ego car?", gt) == "yes"
    assert p4_from_nuscenes("Is there any traffic element in the front view?", gt) is None  # not in nuScenes


def test_correct_traffic_light_is_not_hallucination(tmp_path):
    gt = tmp_path / "gt"
    gt.mkdir()
    json.dump({"CAM_FRONT": []}, open(gt / "s2.json", "w"))  # nuScenes has no box for traffic lights
    preds = [{"qtype": "P1", "sample_token": "s2", "question": "", "answer": "", "answer_label": None,
              "tags": [{"in": "answer", "cam": "CAM_FRONT", "x": 800, "y": 100}],
              "prediction": "<c1,CAM_FRONT,810,105> <c2,CAM_FRONT,300,600>"}]
    r = evaluate(preds, str(gt))
    assert r["hallucination_rate"] == 50.0  # only the unmatched tag c2
    assert r["grounding_rate"] == 50.0


def test_set_image_pixels_handles_both_processor_styles():
    from types import SimpleNamespace
    from dlp.models.compat import set_image_pixels
    new_style = SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 1, "longest_edge": 2}))
    set_image_pixels(new_style, 100, 200)
    assert new_style.image_processor.size == {"shortest_edge": 100, "longest_edge": 200}
    old_style = SimpleNamespace(image_processor=SimpleNamespace(min_pixels=1, max_pixels=2))
    set_image_pixels(old_style, 100, 200)
    assert (old_style.image_processor.min_pixels, old_style.image_processor.max_pixels) == (100, 200)


def test_p2_turn_metrics(tmp_path):
    rows = [{"qtype": "P2", "question": "", "answer": "", "answer_label": lab, "tags": [], "sample_token": "x",
             "prediction": pred} for lab, pred in
            [("going_ahead", "Going ahead.")] * 8 + [("turn_left", "Turn left."), ("turn_right", "Going ahead."),
                                                      ("going_ahead", "Turn right.")]]
    r = evaluate(rows, str(tmp_path))["P2"]
    assert r["n_turn"] == 2 and r["turn_recall"] == 50.0 and r["turn_any_recall"] == 50.0
    assert r["turn_precision"] == 50.0 and r["label_counts"]["going_ahead"] == 9


def test_unanswered_types_are_not_scored(tmp_path):
    rows = [{"qtype": "P3", "question": "", "answer": "", "answer_label": "car", "tags": [], "sample_token": "x",
             "prediction": "A car."},
            {"qtype": "P4", "question": "", "answer": "", "answer_label": "yes", "tags": [], "sample_token": "x",
             "prediction": ""}]
    r = evaluate(rows, str(tmp_path))
    assert "P4" not in r and r["unanswered_qtypes"] == ["P4"] and r["parse_fail_rate"] == 0.0


def test_find_run_matches_experiment_exactly(tmp_path):
    from _common import find_run
    for d in ("1110_r1_10k_d1_s0", "1110_r1_10k_turnx8_d1_s0", "1110_p2full_r1_d1_s0", "1110_p2full_r1_10k_d1_s0"):
        (tmp_path / d / "adapter").mkdir(parents=True)
    assert find_run(str(tmp_path), "r1_10k", "adapter").endswith("1110_r1_10k_d1_s0")
    assert find_run(str(tmp_path), "r1_10k_turnx8").endswith("1110_r1_10k_turnx8_d1_s0")
    assert find_run(str(tmp_path), "p2full_r1").endswith("1110_p2full_r1_d1_s0")


# ---------------------------------------------------------------- bootstrap CIs
def test_bootstrap_ci_contains_point_and_is_deterministic():
    import random
    from dlp.eval.metrics import bootstrap_ci, stat
    rnd = random.Random(0)
    labs = ["moving", "stopped", "turn_left"]
    data = []
    for _ in range(300):
        t = rnd.choice(labs)
        data.append((t, t if rnd.random() < 0.7 else rnd.choice(labs)))
    scenes = [i // 10 for i in range(300)]
    for kind in ("macro", "bacc", "acc"):
        point = stat(kind, data)
        lo, hi = bootstrap_ci(kind, data)
        assert lo <= point <= hi and hi - lo < 25
        assert bootstrap_ci(kind, data) == [lo, hi]
        slo, shi = bootstrap_ci(kind, data, scenes)
        assert slo <= point <= shi
    counts = [(rnd.randint(0, 3), rnd.randint(0, 2), rnd.randint(0, 2)) for _ in range(200)]
    lo, hi = bootstrap_ci("micro", counts)
    assert lo <= stat("micro", counts) <= hi


def test_cluster_bootstrap_is_wider_for_correlated_rows():
    from dlp.eval.metrics import bootstrap_ci
    # 20 scenes x 15 questions; every question of a scene is right or wrong together
    data, scenes = [], []
    for s in range(20):
        ok = s % 3 != 0
        for _ in range(15):
            data.append(("yes", "yes" if ok else "no"))
            scenes.append(s)
    q = bootstrap_ci("acc", data)
    c = bootstrap_ci("acc", data, scenes)
    assert (c[1] - c[0]) > 2 * (q[1] - q[0])


def test_bootstrap_perfect_and_tiny():
    from dlp.eval.metrics import bootstrap_ci
    y = [("yes", "yes"), ("no", "no")] * 20
    assert bootstrap_ci("bacc", y) == [100.0, 100.0]
    assert bootstrap_ci("acc", [("yes", "yes")]) is None
    assert bootstrap_ci("micro", [(1, 0, 0)]) is None


def test_paired_bootstrap_detects_consistent_gain():
    from dlp.eval.metrics import paired_bootstrap
    # B is right on 60% of questions; A is right on those plus 10% more, spread over 30 scenes
    a, b = [], []
    for i in range(300):
        sc = i // 10  # every scene holds one question of each kind
        rb = i % 10 < 6
        ra = rb or i % 10 == 6
        a.append((f"q{i}", ("yes", "yes" if ra else "no"), sc))
        b.append((f"q{i}", ("yes", "yes" if rb else "no"), sc))
    r = paired_bootstrap(("acc", a, None), ("acc", b, None))
    assert r["diff"] == 10.0 and r["ci95"][0] > 0 and r["n"] == 300
    same = paired_bootstrap(("acc", b, None), ("acc", b, None))
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]


def test_few_shot_examples_become_earlier_turns():
    from dlp.data.dataset import build_messages
    row = {"question": "Q?", "answer": "A.", "images": {}}
    m = build_messages(row, ".", "none", with_answer=False, examples=[("q1?", "a1."), ("q2?", "a2.")])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert m[2]["content"][0]["text"] == "a1." and m[-1]["content"][-1]["text"] == "Q?"
    assert len(build_messages(row, ".", "none", with_answer=False)) == 2


def test_road_sign_is_a_traffic_element():  # found in the hand check of 100 scored answers
    from dlp.data.templates import category_from_text
    for t in ("Road sign.", "Street sign.", "Traffic sign.", "A speed limit sign.", "Stop sign."):
        assert category_from_text(t) == "traffic_element", t
    assert category_from_text("A person holding a sign.") == "pedestrian"

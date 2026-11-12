import random

from PIL import Image

from dlp.data.dataset import (IGNORE_INDEX, build_messages, choose_cameras, label_check, make_grid, mask_labels,
                              stratified_sample, apply_oversample)


def test_mask_only_answer_tokens():
    ids = [1, 2, 3, 4, 5, 6, 0, 0]
    labels = mask_labels(ids, prompt_len=4, pad_id=0)
    assert labels == [IGNORE_INDEX] * 4 + [5, 6, IGNORE_INDEX, IGNORE_INDEX]
    rep = label_check(ids, labels, decode=lambda xs: " ".join(map(str, xs)))
    assert rep["n_supervised"] == 2 and rep["supervised_text"] == "5 6"


def test_camera_dropout_keeps_referenced_cameras():
    row = {"tags": [{"cam": "CAM_BACK"}]}
    rng = random.Random(0)
    for _ in range(50):
        cams = choose_cameras(row, dropout=0.9, rng=rng)
        assert "CAM_BACK" in cams
    assert choose_cameras(row, keep_only=["CAM_FRONT"]) == ["CAM_FRONT"]
    assert len(choose_cameras(row, dropout=0.0)) == 6


def test_grid_layout():
    imgs = {"CAM_FRONT": Image.new("RGB", (10, 10), (255, 0, 0)), "CAM_BACK_RIGHT": Image.new("RGB", (10, 10), (0, 255, 0))}
    g = make_grid(imgs, cell=(10, 10))
    assert g.size == (30, 20)
    assert g.getpixel((15, 5)) == (255, 0, 0)      # top middle = CAM_FRONT
    assert g.getpixel((25, 15)) == (0, 255, 0)     # bottom right = CAM_BACK_RIGHT
    assert g.getpixel((5, 5)) == (0, 0, 0)         # missing camera -> black


def test_build_messages_modes(tmp_path):
    row = {"images": {}, "question": "Q?", "answer": "A."}
    sep = build_messages(row, str(tmp_path), "separate", ["CAM_FRONT", "CAM_BACK"])
    user = sep[1]["content"]
    assert [c["type"] for c in user] == ["text", "image", "text", "image", "text"]
    assert user[0]["text"] == "CAM_FRONT:" and sep[-1]["role"] == "assistant"
    grid = build_messages(row, str(tmp_path), "grid", with_answer=False)
    assert sum(c["type"] == "image" for c in grid[1]["content"]) == 1 and grid[-1]["role"] == "user"
    noname = build_messages(row, str(tmp_path), "separate", ["CAM_FRONT"], camera_names=False)
    assert [c["type"] for c in noname[1]["content"]] == ["image", "text"]


def test_stratified_sample_balances_types():
    rows = [{"qtype": "P1", "i": i} for i in range(5)] + [{"qtype": "P4", "i": i} for i in range(200)] + \
           [{"qtype": "P5", "i": i} for i in range(100)]
    out = stratified_sample(rows, 60, seed=0)
    counts = {k: sum(r["qtype"] == k for r in out) for k in ("P1", "P4", "P5")}
    assert len(out) == 60 and counts["P1"] == 5 and counts["P4"] + counts["P5"] == 55
    assert abs(counts["P4"] - counts["P5"]) <= 1


def test_oversample_turns():
    rows = [{"qtype": "P2", "answer_label": "going_ahead"}] * 9 + [{"qtype": "P2", "answer_label": "turn_left"}] + \
           [{"qtype": "P4", "answer_label": "yes"}] * 3
    out = apply_oversample(rows, [{"qtype": "P2", "labels": ["turn_left", "turn_right"], "factor": 8}])
    assert sum(r["answer_label"] == "turn_left" for r in out) == 8 and len(out) == 9 + 8 + 3
    assert apply_oversample(rows, None) is rows

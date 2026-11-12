from dlp.data.linking import iou, link_tag
from dlp.data.templates import Tag
from dlp.eval.matching import ground_tag, match_tags


def t(cam, x, y):
    return {"cam": cam, "x": x, "y": y}


def test_one_to_one_and_threshold():
    gt = [t("CAM_FRONT", 100, 100), t("CAM_FRONT", 130, 100)]
    pred = [t("CAM_FRONT", 115, 100)]  # close to both, may match only one
    m = match_tags(pred, gt, tau=50)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)
    assert match_tags([t("CAM_FRONT", 300, 100)], gt, tau=50)["tp"] == 0


def test_cameras_never_match_across():
    m = match_tags([t("CAM_BACK", 100, 100)], [t("CAM_FRONT", 100, 100)], tau=50)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)


def test_hungarian_beats_greedy():
    gt = [t("CAM_FRONT", 0, 0), t("CAM_FRONT", 60, 0)]
    pred = [t("CAM_FRONT", 35, 0), t("CAM_FRONT", 95, 0)]
    # greedy would pair pred0-gt1 (25px) and leave pred1 unmatched at tau=40; optimal pairs both
    assert match_tags(pred, gt, tau=40)["tp"] == 2


def test_grounding_and_hallucination():
    boxes = {"CAM_FRONT": [{"box2d": [100, 100, 200, 200]}]}
    assert ground_tag(t("CAM_FRONT", 150, 150), boxes)["grounded"]
    near = ground_tag(t("CAM_FRONT", 250, 150), boxes)
    assert not near["grounded"] and not near["hallucinated"] and near["nearest_px"] == 50
    assert ground_tag(t("CAM_FRONT", 600, 600), boxes)["hallucinated"]
    assert ground_tag(t("CAM_BACK", 150, 150), boxes)["hallucinated"]


def test_linking_rules():
    boxes = {"CAM_FRONT": [
        {"ann_token": "big", "box2d": [0, 0, 400, 400], "center": [200, 200]},
        {"ann_token": "small", "box2d": [90, 90, 130, 130], "center": [110, 110]},
    ]}
    assert link_tag(Tag("c1", "CAM_FRONT", 110, 110), boxes)["ann_token"] == "small"
    assert link_tag(Tag("c1", "CAM_FRONT", 110, 110), boxes, bbox2d=[0, 0, 390, 400])["ann_token"] == "big"
    near = link_tag(Tag("c1", "CAM_FRONT", 430, 200), boxes)
    assert near["link"] == "near" and near["ann_token"] == "big"
    assert link_tag(Tag("c1", "CAM_FRONT", 900, 900), boxes) is None
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9

from dlp.data.templates import (
    Tag, categories_from_text, category_from_text, category_group, classify_question, coarse_status,
    count_malformed_tags, format_tag, normalize_status, normalize_yesno, parse_options, parse_p4, parse_tags,
    region_camera, resolve_choice, status_from_nuscenes, status_set, tag_valid,
)


def test_parse_and_format_roundtrip():
    text = "IDs are <c1,CAM_FRONT,1023.3,511.7>, <c2,CAM_BACK_LEFT,12.0,800.25> and again <c1,CAM_FRONT,1023.3,511.7>."
    tags = parse_tags(text)
    assert [t.cid for t in tags] == ["c1", "c2"]  # duplicate removed
    assert tags[1] == Tag("c2", "CAM_BACK_LEFT", 12.0, 800.25)
    assert format_tag(tags[0]) == "<c1,CAM_FRONT,1023.3,511.7>"
    assert parse_tags(format_tag(tags[1]))[0].cam == "CAM_BACK_LEFT"


def test_tag_tolerates_spaces_and_counts_malformed():
    assert parse_tags("< c3 , CAM_FRONT_RIGHT , 5 , 6 >")[0].x == 5.0
    assert count_malformed_tags("<c1,CAM_FRONT,abc,1> <c2,CAM_FRONT,1,2>") == 1


def test_tag_valid():
    assert tag_valid(Tag("c1", "CAM_FRONT", 10, 10))
    assert not tag_valid(Tag("c1", "CAM_TOP", 10, 10))
    assert not tag_valid(Tag("c1", "CAM_FRONT", 1700, 10))


def test_classify_real_v11_templates():
    """Wording copied from DriveLM-nuScenes v1.1 (template counts printed on 2026-11-10)."""
    cases = {
        "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.": "P1",
        "What is the moving status of object <c1,CAM_BACK,1088.3,497.5>?": "P2",
        "What is the observed status of object <c1,CAM_BACK,1088.3,497.5>?": "P2o",
        "What is the visual description of <c1,CAM_BACK,1088.3,497.5>?": "P3",
        "Are there parked cars to the front left of the ego car?": "P4",
        "Are there motorcycles without riders to the back of the ego car?": "P4",
        "Is there any traffic element in the front view?": "P4",
        "What are objects to the front right of the ego car?": "P5",
        "What is the status of the trucks that are to the front left of the ego car?": "P6",
        "What is the status of the truck that is to the back of the ego car?": "P6",
        "What is the relative positioning of the important objects in the current scene?": "other",
        "Which lanes are each important object on in the scene?": "other",
        "Please describe the current scene.": "other",
    }
    for q, want in cases.items():
        assert classify_question(q) == want, q


def test_regions_and_p4_parsing():
    assert region_camera("What are objects to the back right of the ego car?") == "CAM_BACK_RIGHT"
    assert region_camera("Is there any traffic element in the front view?") == "CAM_FRONT"
    assert parse_p4("Are there parked construction vehicles to the back left of the ego car?") == \
        {"attr": "parked", "group": "truck", "nusc": "vehicle.construction", "cam": "CAM_BACK_LEFT"}
    assert parse_p4("Are there parked trucks to the front of the ego car?")["nusc"] == "vehicle.truck"
    assert parse_p4("Are there bicycles without riders to the front of the ego car?")["attr"] == "without_rider"
    assert parse_p4("Are there barriers to the front of the ego car?") == \
        {"attr": None, "group": "barrier", "nusc": "movable_object.barrier", "cam": "CAM_FRONT"}


def test_region_answers():
    a = "There are many barriers and one construction vehicle to the front right of the ego car."
    assert categories_from_text(a) == {"barrier", "truck"}          # 'ego car' and 'vehicle' are not cars
    assert categories_from_text("There are two cars and many pedestrians.") == {"car", "pedestrian"}
    assert status_set("Many trucks are parked.") == "stopped"
    assert status_set("One truck is moving.") == "moving"
    assert status_set("One car is moving, two cars are parked.") == "moving+stopped"
    # found in B2 examples on 2026-11-10
    assert status_set("One of the pedestrians is standing, and three are moving.") == "moving+stopped"
    assert status_set("The pedestrians to the back left of the ego car are standing on the sidewalk.") == "stopped"
    assert status_set("Standing.") == "stopped"
    # cycles: DriveLM asks about the rider (found 2026-11-10, 1,993 train answers)
    assert status_set("One motorcycle is without a rider.") == "no_rider"
    assert status_set("One bicycle is with a rider.") == "rider"
    assert status_set("One bicycle has a rider.") == "rider"
    assert status_set("One bicycle is being ridden.") == "rider"
    assert status_set("Two bicycles are without riders.") == "no_rider"
    assert status_set("One of the bicycles has a rider, and one does not have a rider.") == "no_rider+rider"
    assert normalize_yesno("There are many barriers.") is None       # not a yes/no answer


def test_options_and_choice():
    q = "What is the moving status? Please select the correct answer from the following options: A. Going ahead. B. Turn left. C. Turn right. D. Stopped."
    opts = parse_options(q)
    assert opts == {"A": "Going ahead.", "B": "Turn left.", "C": "Turn right.", "D": "Stopped."}
    assert resolve_choice("D.", opts) == "Stopped."
    assert resolve_choice("B. Turn left.", opts) == "Turn left."
    assert resolve_choice("Going ahead.", opts) == "Going ahead."


def test_status_and_category_normalisation():
    assert normalize_status("The car is turning left") == "turn_left"
    assert normalize_status("Stationary.") == "stopped"
    assert normalize_status("Going ahead.") == "going_ahead"
    assert coarse_status("turn_right") == "moving"
    assert status_from_nuscenes("vehicle.parked", None) == "stopped"
    assert status_from_nuscenes(None, 3.0) == "moving"
    assert status_from_nuscenes("vehicle.stopped", 0.0) == "stopped"   # DriveLM calls these 'Stationary'
    assert status_from_nuscenes("cycle.with_rider", 0.1) == "stopped"  # rider attribute: speed decides
    assert category_from_text("White SUV.") == "car"
    # cases found by the 2026-11-10 diagnosis on the real data
    assert category_from_text("Vehicle White van.") == "car"                     # nuScenes: vans are cars
    assert category_from_text("Cycle Person riding a motorcycle.") == "motorcycle"
    assert category_from_text("Cycle Person riding a bicycle.") == "bicycle"
    assert category_from_text("Vehicle White commercial vehicle.") == "vehicle"   # unspecific
    assert category_from_text("A bus parked") == "bus"
    assert category_from_text("Pedestrian in black.") == "pedestrian"
    assert category_group("human.pedestrian.adult") == "pedestrian"
    assert category_group("vehicle.bus.rigid") == "bus"
    assert normalize_yesno("Yes, there is one.") == "yes"
    assert normalize_yesno("No.") == "no"

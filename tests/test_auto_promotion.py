from typing import Any

import numpy as np
import pytest
from maa.define import OCRResult, RecognitionDetail, Rect

from agent.custom.reco.auto_promotion import APMapAnalyze, StageEntry, is_stage_map


def _detail(items: list[tuple[str, float, list[int]]], hit: bool = True) -> RecognitionDetail:
    results = [OCRResult(Rect(*box), score, text) for text, score, box in items]
    return RecognitionDetail(
        reco_id=1,
        name="test",
        algorithm="OCR",
        hit=hit,
        box=Rect(0, 0, 0, 0),
        all_results=results,
        filtered_results=results,
        best_result=results[0] if results else None,
        raw_detail={},
        raw_image=np.zeros((1, 1, 3), dtype=np.uint8),
        draw_images=[],
    )


class _FakeContext:
    def __init__(self, details: dict[str, RecognitionDetail]) -> None:
        self.details = details

    def run_recognition(self, name: str, image: Any, *args: Any, **kwargs: Any) -> RecognitionDetail:
        return self.details.get(name, _detail([], hit=False))


def _image() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_parse_stage_number_variants() -> None:
    analyzer = APMapAnalyze()
    box = [100, 540, 80, 40]
    expected = {
        "A051FH": 5,
        "051F": 5,
        "051E": 5,
        "13iES": 13,
        "04/E": 4,
        "08/F": 8,
        "A1333": 13,
        "04": 4,
        "08": 8,
    }
    for text, number in expected.items():
        parsed = analyzer._parse_stage_number(text, box)
        assert parsed is not None, text
        assert parsed[0] == number, text
    assert analyzer._parse_stage_number("00", box) is None
    assert analyzer._parse_stage_number("70", box) is None


def test_stage_name_filter_rejects_ui_text() -> None:
    analyzer = APMapAnalyze()
    assert analyzer._is_valid_stage_name("旧诫言")
    assert analyzer._is_valid_stage_name("坏钟表")
    assert analyzer._is_valid_stage_name("自由的频段")
    assert not analyzer._is_valid_stage_name("开始行动")
    assert not analyzer._is_valid_stage_name("推荐等级")
    assert not analyzer._is_valid_stage_name("故事模式")
    assert not analyzer._is_valid_stage_name("1/3 12:42")
    assert not analyzer._is_valid_stage_name("F/80")
    assert not analyzer._is_valid_stage_name("08")
    assert analyzer._is_stage_name_box([556, 579, 97, 39])
    assert not analyzer._is_stage_name_box([0, 590, 200, 11])  # 英文副标题字号过小


def test_stage_entries_pair_name_and_number() -> None:
    context = _FakeContext(
        {
            "APStageNameOCR": _detail(
                [
                    ("危险席位", 1.0, [370, 574, 125, 41]),
                    ("旧诫言", 0.99, [556, 579, 97, 39]),
                ]
            ),
            "APStageNumberOCR": _detail(
                [
                    ("13iES", 0.83, [369, 540, 135, 42]),
                    ("08/", 0.89, [551, 547, 101, 37]),
                ]
            ),
        }
    )
    entries = APMapAnalyze()._stage_entries(context, _image())
    by_name = {entry.name: entry for entry in entries}
    assert by_name["危险席位"].number == 13
    assert by_name["危险席位"].number_box == [369, 540, 135, 42]
    assert by_name["旧诫言"].number == 8
    assert [entry.name for entry in entries] == ["旧诫言", "危险席位"] or [
        entry.name for entry in entries
    ] == ["危险席位", "旧诫言"]


def test_name_probe_excludes_current_stage_marker() -> None:
    analyzer = APMapAnalyze()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    # 当前关卡金色三角：位于名称框右侧 780-820px，不应进入名称探针
    image[560:600, 780:820] = (0, 200, 255)
    name_box = [556, 579, 97, 39]
    probe = analyzer._name_probe_box(name_box, 1280, 720)
    assert probe[2] > 0 and probe[3] > 0
    assert analyzer._lit_count_box(image, probe) == 0
    # 探针内部放一颗亮星，验证计数路径没有坏
    image[585:605, 690:710] = (0, 220, 255)
    assert analyzer._lit_count_box(image, probe) > 0


def test_stage_entries_name_only_without_number() -> None:
    context = _FakeContext(
        {
            "APStageNameOCR": _detail([("坏钟表", 1.0, [473, 577, 91, 38])]),
            "APStageNumberOCR": _detail([], hit=False),
        }
    )
    entries = APMapAnalyze()._stage_entries(context, _image())
    assert len(entries) == 1
    assert entries[0].name == "坏钟表"
    assert entries[0].number is None
    assert entries[0].source == "name"


def test_is_stage_map_accepts_name_entries_when_start_action_visible() -> None:
    context = _FakeContext(
        {
            "APStageNameOCR": _detail([("旧诫言", 1.0, [556, 579, 97, 39])]),
            "APStageNumberOCR": _detail([("08", 1.0, [551, 547, 101, 37])]),
            "AP_StartAction": _detail([("开始行动", 1.0, [1090, 587, 126, 38])]),
        }
    )
    assert is_stage_map(context, _image()) is False
    assert is_stage_map(context, _image(), allow_start_action=True) is True


def test_empty_stage_entries_never_done() -> None:
    context = _FakeContext(
        {
            "APStageNameOCR": _detail([], hit=False),
            "APStageNumberOCR": _detail([], hit=False),
        }
    )
    analyzer = APMapAnalyze()
    result = analyzer.analyze(
        context,
        type("Arg", (), {"image": _image(), "custom_recognition_param": '{"query":"done"}'})(),
    )
    assert result is None


def test_end_detection_requires_stable_name_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _FakeContext(
        {
            "APStageNameOCR": _detail([("旧诫言", 1.0, [556, 579, 97, 39])]),
            "APStageNumberOCR": _detail([("08", 1.0, [551, 547, 101, 37])]),
        }
    )
    analyzer = APMapAnalyze()
    analyzer._stage_complete_entry = lambda image, entry: (True, "ok")  # type: ignore[method-assign]
    APMapAnalyze.reset_swipe_state()
    arg = type("Arg", (), {"image": _image(), "custom_recognition_param": '{"query":"swipe"}'})()
    assert analyzer.analyze(context, arg) is not None
    assert analyzer.analyze(context, arg) is not None
    assert analyzer.analyze(context, arg) is None
    done_arg = type("Arg", (), {"image": _image(), "custom_recognition_param": '{"query":"done"}'})()
    assert analyzer.analyze(context, done_arg) is not None
    APMapAnalyze.reset_swipe_state()


def test_click_box_prefers_clean_number_and_expands_wide_number_to_name() -> None:
    analyzer = APMapAnalyze()
    clean = StageEntry("旧诫言", 8, [556, 579, 97, 39], [551, 547, 64, 37], 604, "both")
    clean_box = analyzer._click_box(clean, 1280, 720)
    assert clean_box[0] == 531  # number box x-20
    wide = StageEntry("旧诫言", 8, [556, 579, 97, 39], [551, 547, 101, 37], 604, "both")
    wide_box = analyzer._click_box(wide, 1280, 720)
    assert wide_box[0] == 536  # name box x-20


def test_legacy_number_path_without_names() -> None:
    context = _FakeContext(
        {
            "APStageNameOCR": _detail([], hit=False),
            "APStageNumberOCR": _detail([("04", 1.0, [188, 549, 64, 37])]),
        }
    )
    analyzer = APMapAnalyze()
    entries = analyzer._stage_entries(context, _image())
    assert len(entries) == 1
    assert entries[0].name is None
    assert entries[0].number == 4
    assert not analyzer._use_name_anchor(entries)

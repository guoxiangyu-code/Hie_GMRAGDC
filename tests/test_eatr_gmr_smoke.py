from methods.eatr_gmr.dataset import video_id_to_feature_stem
from methods.eatr_gmr.smoke import run_smoke


def test_video_extension_is_removed():
    assert video_id_to_feature_stem("match_001.mp4") == "match_001"
    assert video_id_to_feature_stem("match_001.MP4") == "match_001"
    assert video_id_to_feature_stem("match_001") == "match_001"


def test_eatr_positive_mixed_and_all_null_smoke():
    report = run_smoke()
    assert report["status"] == "ok"
    assert set(report["variants"]) == {
        "eatr", "eatr_gmr", "eatr_quality", "eatr_dual",
        "eatr_quality_dual", "eatr_counter", "eatr_hiea2m",
    }
    for variant in report["variants"].values():
        assert set(variant["batches"]) == {"positive", "mixed", "all_null"}
    assert not report["variants"]["eatr"]["official_evaluator"]["has_explicit_exist_score"]
    assert report["variants"]["eatr_gmr"]["official_evaluator"]["has_explicit_exist_score"]
    assert not report["variants"]["eatr_quality_dual"]["official_evaluator"]["has_count_metrics"]
    assert report["variants"]["eatr_counter"]["official_evaluator"]["has_count_metrics"]
    assert report["variants"]["eatr_hiea2m"]["official_evaluator"]["has_count_metrics"]

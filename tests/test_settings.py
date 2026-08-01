"""Tests for persistent analyzer-wide preferences."""

import json

from orca_nics_analyzer.settings import DEFAULT_SETTINGS, load_settings, save_settings


def test_settings_round_trip_preserves_unrelated_sections(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"other_plugin": {"enabled": True}}), encoding="utf-8")

    values = dict(DEFAULT_SETTINGS)
    values["map_levels"] = 47
    values["icss_opacity"] = 0.8
    assert save_settings(values, str(path))

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["other_plugin"] == {"enabled": True}
    assert saved["nics_analyzer_settings"]["map_levels"] == 47
    assert saved["nics_analyzer_settings"]["icss_opacity"] == 0.8

    loaded = load_settings(str(path))
    assert loaded["map_levels"] == 47
    assert loaded["icss_opacity"] == 0.8


def test_corrupt_settings_fall_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_settings(str(path)) == DEFAULT_SETTINGS


def test_partial_settings_only_override_known_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "nics_analyzer_settings": {
                    "map_levels": 55,
                    "unknown_setting": "ignored",
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_settings(str(path))
    assert loaded["map_levels"] == 55
    assert "unknown_setting" not in loaded
    assert loaded["map_auto_range"] is True

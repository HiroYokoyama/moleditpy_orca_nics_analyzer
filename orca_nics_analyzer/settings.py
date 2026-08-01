"""Persistent user preferences for the ORCA NICS Analyzer plugin."""

import json
import logging
import os

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
SETTINGS_KEY = "nics_analyzer_settings"

DEFAULT_SETTINGS = {
    "axis_mode": "grid",
    "axis_vector": [0.0, 0.0, 1.0],
    "show_probes": False,
    "map_component": "zz",
    "map_colormap": "seismic",
    "map_levels": 31,
    "map_range": 10.0,
    "map_auto_range": True,
    "map_molecule": True,
    "map_contours": True,
    "map_probes": False,
    "map_slice_line": False,
    "icss_component": "zz",
    "icss_colormap": "seismic",
    "icss_isovalue": 5.0,
    "icss_opacity": 0.55,
    "icss_positive": True,
    "icss_negative": True,
    "icss_cut_axis": False,
    "icss_show_vector": False,
}


def load_settings(path=SETTINGS_FILE):
    """Load validated, molecule-independent preferences from *path*."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        values = data.get(SETTINGS_KEY, {}) if isinstance(data, dict) else {}
        if isinstance(values, dict):
            for key in settings:
                if key in values:
                    settings[key] = values[key]
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        if os.path.exists(path):
            logging.warning("[orca_nics_analyzer] load settings: %s", exc)
    return settings


def save_settings(settings, path=SETTINGS_FILE):
    """Atomically save preferences while preserving other JSON sections."""
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            data = existing
    except (OSError, TypeError, ValueError):
        pass

    data[SETTINGS_KEY] = {
        key: settings[key] for key in DEFAULT_SETTINGS if key in settings
    }
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            os.remove(temporary)
        except OSError:
            pass
        logging.warning("[orca_nics_analyzer] save settings: %s", exc)
        return False
    return True

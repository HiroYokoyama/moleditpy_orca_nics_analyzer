"""Persistent user preferences for the ORCA NICS Analyzer plugin.

Only molecule-independent choices belong here — colormaps, which overlays
are on, how many contour levels. Two kinds of value are deliberately left
out because they only mean something for the structure they were picked on:

* The NICS_zz axis (mode and manual vector). Lab X/Y/Z and a hand-typed
  direction are relative to one geometry's own orientation, so carrying
  them over would project the next file onto a direction the user never
  chose for it. Every load starts from the ICSS convention.
* The ppm scales (map range, ICSS isovalue). Both are sized from the data
  itself, and a threshold that frames one molecule's ring current can be
  orders of magnitude wrong for the next.
"""

import json
import logging
import os

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
SETTINGS_KEY = "nics_analyzer_settings"

#: What the NICS_zz axis resets to for every newly loaded output.
DEFAULT_AXIS_MODE = "grid"

DEFAULT_SETTINGS = {
    "show_probes": False,
    "map_component": "zz",
    "map_colormap": "seismic",
    "map_levels": 31,
    "map_molecule": True,
    "map_contours": True,
    "map_probes": False,
    "map_slice_line": False,
    "icss_component": "zz",
    "icss_colormap": "seismic",
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

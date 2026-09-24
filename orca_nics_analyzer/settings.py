"""Persistent user preferences for the ORCA NICS Analyzer plugin.

The line is between what the *user* chose and what the *program* computed.
A deliberate choice is a preference and belongs here even when it is in
ppm — comparing a series of molecules on one fixed colour range is exactly
how ICSS maps get read. A number the code derived from one structure's own
data is not a preference, and is left out:

* The NICS_zz axis (mode and manual vector). Lab X/Y/Z and a hand-typed
  direction are relative to one geometry's own orientation, so carrying
  them over would project the next file onto a direction the user never
  chose for it. Every load starts from the ICSS convention.
* The ICSS isovalue, which is sized from the data at load: a threshold
  framing one molecule's ring current can be orders of magnitude wrong for
  the next.

``map_range`` is stored, but only ever the value the user typed — the
auto-computed span must not overwrite the remembered preference.
"""

import json
import logging
import math
import os

logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
SETTINGS_KEY = "nics_analyzer_settings"

#: What the NICS_zz axis resets to for every newly loaded output.
DEFAULT_AXIS_MODE = "grid"

DEFAULT_SETTINGS = {
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
    "icss_opacity": 0.55,
    "icss_positive": True,
    "icss_negative": True,
    "icss_cut_axis": False,
    "icss_show_vector": False,
}


def _valid(value, default):
    """True when *value* has the type of its default, so the dialog can use it.

    A hand-edited or corrupted file would otherwise crash the dialog on open
    (``int("abc")``) or tick a checkbox from a string.
    """
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    return isinstance(value, type(default))


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
                    if _valid(values[key], DEFAULT_SETTINGS[key]):
                        settings[key] = values[key]
                    else:
                        logger.warning(
                            "[orca_nics_analyzer] ignoring bad setting %s=%r",
                            key,
                            values[key],
                        )
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        if os.path.exists(path):
            logger.warning("[orca_nics_analyzer] load settings: %s", exc)
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
        logger.warning("[orca_nics_analyzer] save settings: %s", exc)
        return False
    return True

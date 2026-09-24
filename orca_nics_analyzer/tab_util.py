"""Helpers shared by the analyzer's tab widgets."""

from PyQt6.QtWidgets import QTabWidget


def is_current_tab(widget):
    """True when *widget* is the page on show in its QTabWidget.

    A page sits in the tab widget's internal QStackedWidget, so the tab widget
    is two parents up. A widget used outside any tab widget counts as shown,
    which is what lets a tab be built and exercised on its own.
    """
    stack = widget.parentWidget()
    tabs = stack.parentWidget() if stack is not None else None
    if isinstance(tabs, QTabWidget):
        return tabs.currentWidget() is widget
    return True

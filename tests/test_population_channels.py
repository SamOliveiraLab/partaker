import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _image_data(channels):
    """Minimal stand-in exposing .data.shape == (T, P, C, Y, X)."""
    obj = types.SimpleNamespace()
    obj.data = np.zeros((6, 1, channels, 32, 32), dtype=np.uint16)
    return obj


def _widget(qapp):
    from partaker.ui.widgets.population import PopulationWidget

    return PopulationWidget()


def test_single_channel_dataset_does_not_crash_on_load(qapp):
    """Channel 0 is phase contrast, so a 1-channel file has no fluorescence
    channel. Loading one must not raise (previously: int('') ValueError)."""
    w = _widget(qapp)
    w.on_image_data_loaded(_image_data(channels=1))
    assert w.mcherry_channel_combo.currentText() == ""


def test_single_channel_plot_button_does_not_crash(qapp):
    w = _widget(qapp)
    w.on_image_data_loaded(_image_data(channels=1))
    w.on_plot_population_signal()


def test_two_channel_dataset_populates_fluorescence_channel(qapp):
    w = _widget(qapp)
    w.on_image_data_loaded(_image_data(channels=2))
    assert w.mcherry_channel_combo.currentText() == "1"
    assert w.yfp_channel_combo.currentText() == "1"

import numpy as np
import pytest
import tifffile

from partaker.data.image_data import ImageData


def _write(tmp_path, name, arr, **kwargs):
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, **kwargs)
    return str(p)


def test_normalize_axes_pads_to_5d():
    arr = np.zeros((5, 64, 64), dtype=np.uint16)
    out, axes = ImageData.normalize_axes(arr, "TYX")
    assert out.shape == (5, 1, 1, 64, 64)
    assert axes == "TPCYX"


def test_normalize_axes_treats_unlabelled_sequence_as_time():
    """Plain multi-page TIFFs report a generic axis ('Q'); it must read as time."""
    arr = np.zeros((5, 64, 64), dtype=np.uint16)
    out, axes = ImageData.normalize_axes(arr, "QYX")
    assert out.shape == (5, 1, 1, 64, 64)
    assert axes == "TPCYX"


def test_normalize_axes_maps_samples_to_channel():
    arr = np.zeros((64, 64, 3), dtype=np.uint16)
    out, axes = ImageData.normalize_axes(arr, "YXS")
    assert out.shape == (1, 1, 3, 64, 64)
    assert axes == "TPCYX"


def test_normalize_axes_reorders_non_canonical_axes():
    """Axes may be stored in any order; they must be transposed, not just padded."""
    arr = np.zeros((2, 3, 8, 9), dtype=np.uint16)  # C, T, Y, X
    out, axes = ImageData.normalize_axes(arr, "CTYX")
    assert out.shape == (3, 1, 2, 8, 9)  # T, P, C, Y, X
    assert axes == "TPCYX"


def test_normalize_axes_preserves_nd2_canonical_order():
    arr = np.zeros((3, 2, 2, 8, 9), dtype=np.uint16)
    out, axes = ImageData.normalize_axes(arr, "TPCYX")
    assert out.shape == (3, 2, 2, 8, 9)
    assert axes == "TPCYX"


def test_normalize_axes_requires_image_plane():
    arr = np.zeros((5, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="lack a X axis"):
        ImageData.normalize_axes(arr, "TY")


def test_normalize_axes_rejects_zstack_with_actionable_message():
    arr = np.zeros((5, 64, 64), dtype=np.uint16)
    with pytest.raises(ValueError, match="Z-stacks are not supported"):
        ImageData.normalize_axes(arr, "ZYX")


def test_normalize_axes_rejects_ambiguous_sequence():
    arr = np.zeros((2, 5, 64, 64), dtype=np.uint16)
    with pytest.raises(ValueError, match="Ambiguous"):
        ImageData.normalize_axes(arr, "TQYX")


@pytest.mark.parametrize(
    "name,arr,kwargs,expected",
    [
        ("single.tif", np.zeros((64, 64), np.uint16), {}, (1, 1, 1, 64, 64)),
        # Plain multi-page stack: no axes metadata at all.
        ("plain.tif", np.zeros((5, 64, 64), np.uint16), {}, (5, 1, 1, 64, 64)),
        (
            "imagej.tif",
            np.zeros((5, 64, 64), np.uint16),
            {"imagej": True, "metadata": {"axes": "TYX"}},
            (5, 1, 1, 64, 64),
        ),
        (
            "ome.tif",
            np.zeros((5, 2, 64, 64), np.uint16),
            {"ome": True, "metadata": {"axes": "TCYX"}},
            (5, 1, 2, 64, 64),
        ),
    ],
)
def test_load_image_accepts_common_tiff_layouts(tmp_path, name, arr, kwargs, expected):
    path = _write(tmp_path, name, arr, **kwargs)
    inst = ImageData.load_image([path])
    assert tuple(inst.data.shape) == expected

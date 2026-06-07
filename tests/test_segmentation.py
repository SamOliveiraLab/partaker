import numpy as np

from partaker.analysis.segmentation.segmentation_models import (
    SegmentationModels,
    preprocess_image,
)


def test_preprocess_returns_float_in_unit_range():
    img = np.random.randint(0, 4096, size=(64, 64)).astype(np.uint16)
    out = preprocess_image(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_constant_image_is_zero():
    img = np.full((32, 32), 100, dtype=np.uint16)
    out = preprocess_image(img)
    assert np.all(out == 0.0)


def test_segmentation_models_is_singleton():
    assert SegmentationModels() is SegmentationModels()


def test_remove_artifacts_drops_small_regions():
    sm = SegmentationModels()
    mask = np.zeros((50, 50), dtype=np.uint16)
    mask[5:25, 5:25] = 1
    mask[0:2, 0:2] = 2
    cleaned = sm.remove_artifacts_from_mask(mask, min_area_ratio=0.2)
    assert 1 in np.unique(cleaned)
    assert 2 not in np.unique(cleaned)


def test_erosion_preserves_labels_and_shrinks_area():
    sm = SegmentationModels()
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[10:30, 10:30] = 7
    out = sm.apply_morphological_erosion([mask], kernel_size=3)
    assert isinstance(out, list) and len(out) == 1
    assert 7 in np.unique(out[0])
    assert (out[0] == 7).sum() < (mask == 7).sum()

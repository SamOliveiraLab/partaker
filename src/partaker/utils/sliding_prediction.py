import numpy as np

# ---------------------------------------------------------------------------
# Sliding-window tiling helpers
# ---------------------------------------------------------------------------


def _compute_tiles(
    image_h: int,
    image_w: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """
    Return a list of ``(y0, x0, y1, x1)`` tile coordinates that together
    cover the full ``(image_h, image_w)`` extent.

    The last tile in each axis is snapped so it always fits inside the image
    (no zero-padding needed), which means the overlap on the far edge may be
    slightly larger than ``overlap``.

    Parameters
    ----------
    image_h, image_w : int
        Spatial dimensions of the image.
    tile_h, tile_w : int
        Tile size — normally the network's training resolution (e.g. 512x512).
    overlap : int
        Number of pixels to overlap between adjacent tiles on each side.
        Must be < tile_h and < tile_w.

    Returns
    -------
    list of (y0, x0, y1, x1)
    """
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    ys = list(range(0, image_h - tile_h + 1, stride_h))
    if not ys or ys[-1] + tile_h < image_h:
        ys.append(max(image_h - tile_h, 0))

    xs = list(range(0, image_w - tile_w + 1, stride_w))
    if not xs or xs[-1] + tile_w < image_w:
        xs.append(max(image_w - tile_w, 0))

    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append((y0, x0, y0 + tile_h, x0 + tile_w))
    return tiles


def _build_weight_map(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """
    Build a 2-D float32 weight map for one tile using a linear ramp in the
    overlap region and 1.0 in the centre.

    Parameters
    ----------
    tile_h, tile_w : int
        Tile spatial dimensions.
    overlap : int
        Overlap region width in pixels.

    Returns
    -------
    np.ndarray
        Shape ``(tile_h, tile_w)``, dtype float32.
    """
    ramp_h = np.ones(tile_h, dtype=np.float32)
    ramp_w = np.ones(tile_w, dtype=np.float32)

    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        ramp_h[:overlap] = ramp
        ramp_h[-overlap:] = ramp[::-1]
        ramp_w[:overlap] = ramp
        ramp_w[-overlap:] = ramp[::-1]

    return np.outer(ramp_h, ramp_w)


def sliding_window_predict(
    model,  # UNetSegmentation instance
    image: np.ndarray,  # (H, W) or (H, W, 1) float32 in [0, 1]
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Run model inference over a full-resolution image using overlapping tiles,
    then stitch the results back into a single probability map.

    Tiles are blended with a linear-ramp weight map so hard seam artefacts at
    tile boundaries are suppressed.

    Parameters
    ----------
    model : UNetSegmentation
        Loaded PyTorch segmentation model.
    image : np.ndarray
        Single image, shape ``(H, W)`` or ``(H, W, 1)``, dtype float32.
    tile_size : int
        Spatial size of each square tile fed to the network (must match the
        resolution the network was trained at, typically 512).
    overlap : int
        Overlap in pixels between neighbouring tiles.  Larger values reduce
        seam artefacts but increase compute.  Typical range: 32-128.

    Returns
    -------
    np.ndarray
        Probability map, shape ``(H, W)``, dtype float32 in ``[0, 1]``.
    """
    # Normalise to (H, W)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")

    h, w = image.shape

    # If the image is smaller than one tile on either axis, just run directly.
    if h <= tile_size and w <= tile_size:
        padded = np.zeros((tile_size, tile_size), dtype=np.float32)
        padded[:h, :w] = image
        inp = padded[np.newaxis, ..., np.newaxis]  # (1, tile_size, tile_size, 1)
        pred = model.predict(inp)[0, :h, :w, 0]  # (H, W)
        return pred

    accumulator = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    tile_weight = _build_weight_map(tile_size, tile_size, overlap).astype(np.float64)

    tiles = _compute_tiles(h, w, tile_size, tile_size, overlap)

    # Batch tiles for efficiency — feed them all at once
    tile_imgs = np.stack(
        [image[y0:y1, x0:x1] for y0, x0, y1, x1 in tiles],
        axis=0,
    )[..., np.newaxis]  # (N_tiles, tile_size, tile_size, 1)

    """
    # TODO: use gpu
    if os.environ("PARTAKER_GPU") == 1:
        # TODO: device provider
        tile_imgs = torch.to(device)
        predictions = model.predict(tile_imgs)  # (N_tiles, tile_size, tile_size, 1)
        pass
    else:
        pass
        """
    predictions = model.predict(tile_imgs)  # (N_tiles, tile_size, tile_size, 1)

    for (y0, x0, y1, x1), pred_tile in zip(tiles, predictions):
        p = pred_tile[..., 0].astype(np.float64)  # (tile_size, tile_size)
        accumulator[y0:y1, x0:x1] += p * tile_weight
        weight_sum[y0:y1, x0:x1] += tile_weight

    # Avoid division by zero for any pixel that was never covered
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
    return (accumulator / weight_sum).astype(np.float32)

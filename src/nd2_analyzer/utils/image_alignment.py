import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import diff
from skimage import exposure


# Shifting the image by a margin of pixels
# Image Analysis
# from aicsimageio import AICSImage


# Dirty Implementation of Shifting Images


def ShiftedImage_2D(Image, XShift, YShift):
    # Quick guard
    if XShift == 0 and YShift == 0:
        return Image

    M = np.float32([
        [1, 0, XShift],
        [0, 1, YShift]
    ])

    shifted = cv2.warpAffine(Image, M, (Image.shape[1], Image.shape[0]))
    shifted_image = shifted

    # Shift Down
    if YShift > 0:
        shifted_image = shifted_image[YShift:]
        shifted_image = np.pad(
            shifted_image, ((YShift, 0), (0, 0)), 'edge')  # Pad Up

    # Shift Up
    if YShift < 0:
        shifted_image = shifted_image[:shifted.shape[0] - abs(YShift)]
        shifted_image = np.pad(
            shifted_image, ((0, abs(YShift)), (0, 0)), 'edge')  # Pad Down

    # Shift Left
    if XShift > 0:
        shifted_image = np.delete(shifted_image, slice(0, XShift), 1)
        shifted_image = np.pad(
            shifted_image, ((0, 0), (XShift, 0)), 'edge')  # Pad Left

    if XShift < 0:
        shifted_image = np.delete(
            shifted_image,
            slice(
                shifted.shape[1] -
                abs(XShift),
                shifted.shape[1]),
            1)
        shifted_image = np.pad(
            shifted_image, ((0, 0), (0, abs(XShift))), 'edge')  # Pad Right

    return shifted_image


def SAD(A, B):
    cutA = A.ravel()
    cutB = B.ravel()
    MAE = np.sum(
        np.abs(
            np.subtract(
                cutA,
                cutB,
                dtype=np.float64))) / cutA.shape[0]
    return MAE


# sum of absolute differences (SAD) metric alignment, quick n dirty
# We use a Tree Search Algorithm to find possible alignment
# Let Image_1 be the orginal
# Let Image_2 be the aligned
# Displacement object is our nodes, [x,y]
# Assumption, there is always a better alignment up, down, left, and right
# if its not the same image


def alignment_MAE(Image_1, Image_2, depth_cap):
    iterative_cap = 0
    Best_SAD = SAD(Image_1, Image_2)
    Best_Displacement = [0, 0]
    q = []
    visited_states = [[0, 0]]  # Add (0,0) displacement
    q.append(Best_Displacement)  # Append (0,0) displacement

    while iterative_cap != depth_cap and q:
        curr_state = q.pop(0)
        x = curr_state[0]
        y = curr_state[1]

        iterative_cap += 1

        movement_arr = [
            [x, y - 1],  # Up
            [x, y + 1],  # Down
            [x + 1, y],  # Left
            [x - 1, y],  # Right
            [x - 1, y - 1],  # Diagonal
            [x + 1, y + 1],  # Diagonal
            [x + 1, y - 1],  # Diagonal
            [x - 1, y + 1],  # Diagonal
        ]

        for move in movement_arr:
            if move not in visited_states:
                visited_states.append(move)  # Marked as Visited

                # Perform shift and calculate
                new_image = ShiftedImage_2D(Image_2, move[0], move[1])
                cand_SAD = SAD(Image_1, new_image)

                if cand_SAD < Best_SAD:
                    Best_SAD = cand_SAD
                    Best_Displacement = move

                    q.append(move)

                # This means we cannot find a better move.

    return Best_Displacement, Best_SAD


# Vec4f is (x1, y1, x2, y2)


# Takes in image and returns the edges for top and bottom parametrically, (x,y)
# Takes in RGB image


# Takes in image and returns the edges for top and bottom parametrically, (x,y)
# Assumes Bottom is always min and top is always max


def edge_cropping_estimation_vertical_high_low_distr(img):
    main_bright = img

    local_vertical = []

    # Vertical Cutting
    for row in range(0, main_bright.shape[0]):
        temp_arr = []
        for col in range(0, main_bright.shape[1]):
            temp_arr.append(np.mean(main_bright[row][col]))
        local_vertical.append(np.mean(temp_arr))

    # ================ Vertical axis squish ================
    x_vertical = list(range(1, main_bright.shape[0] + 1))
    y_vertical = local_vertical

    dydx_vertical = diff(y_vertical) / diff(x_vertical)
    y_verticle_dydx = list(range(1, main_bright.shape[0]))

    for i in range(0, len(dydx_vertical)):
        # Below Crazy 150 values
        if ((dydx_vertical[i] >= 150 and i <= 100) or (
                dydx_vertical[i] <= -150 and i <= 100)):
            dydx_vertical[i] = 0

        # Above Crazy 150 values
        if ((dydx_vertical[i] >= 150 and i >= (main_bright.shape[0] - 100))
                or (dydx_vertical[i] <= -150 and (main_bright.shape[0] - 100))):
            dydx_vertical[i] = 0

    max_val = np.max(dydx_vertical)
    max_index = np.where(dydx_vertical == max_val)[0][0]
    while max_index > (img.shape[1] / 2):
        # print("Cycling max_index:", max_index)
        # Reset the value as it is not needed anymore
        dydx_vertical[max_index] = 0
        max_val = np.max(dydx_vertical)
        max_index = np.where(dydx_vertical == max_val)[0][0]

    min_val = np.min(dydx_vertical)
    min_index = np.where(dydx_vertical == min_val)[0][0]
    while min_index < (img.shape[1] / 2):
        # print("Cycling min_index:", min_index)
        # Reset the value as it is not needed anymore
        dydx_vertical[min_index] = 0
        min_val = np.min(dydx_vertical)
        min_index = np.where(dydx_vertical == min_val)[0][0]

    # print("The VERTICAL DERIVATIVE (Pattern Distribution):")
    plt.plot(y_verticle_dydx, dydx_vertical)
    plt.axvline(x=max_index, color='r')
    plt.axvline(x=min_index, color='r')
    plt.show()

    top = max_index
    bottom = min_index

    return top, bottom


def process_image_sequence_memmap(input_path, output_path, depth_cap=50, m=5, batch_size=10):
    """
    Processes large image sequences using memory-mapped arrays
    Args:
        input_path: Path to memory-mapped array file (.npy)
        output_path: Output file path for aligned images
        depth_cap: Alignment search depth
        batch_size: Number of images to process at once
    Returns:
        Memmap array handle for aligned images
    """
    # Create memory maps for input/output
    arr = np.load(input_path, mmap_mode='r')
    aligned_shape = (arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])

    # Create output memmap
    aligned_arr = np.lib.format.open_memmap(
        output_path,
        dtype=arr.dtype,
        mode='w+',
        shape=aligned_shape
    )

    # Process in batches to balance memory/performance
    for batch_start in range(0, arr.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, arr.shape[0])

        # Load batch with memory mapping
        batch = arr[batch_start:batch_end]
        aligned_batch = aligned_arr[batch_start:batch_end]

        # Process first image as reference
        if batch_start == 0:
            base_image = exposure.rescale_intensity(batch[0, ..., 0])  # Use first channel
            base_top, base_bottom = edge_cropping_estimation_vertical_high_low_distr(base_image)
            aligned_batch[0] = batch[0]
            continue

        # Process subsequent images
        for i in range(batch_start == 0, batch.shape[0]):
            current_img = exposure.rescale_intensity(batch[i, ..., 0])

            # Calculate alignment
            displacement, score = alignment_MAE(
                base_image,
                current_img,
                depth_cap
            )

            # Get vertical shift from edge detection
            current_top, current_bottom = edge_cropping_estimation_vertical_high_low_distr(current_img)
            y_shift = int(np.mean([
                (base_top - current_top),
                (base_bottom - current_bottom)
            ]))

            # Apply shifts to all channels
            for c in range(arr.shape[3]):
                aligned_batch[i, ..., c] = ShiftedImage_2D(
                    batch[i, ..., c],
                    displacement[0],
                    y_shift
                )

        # Flush changes to disk
        aligned_arr.flush()

    return aligned_arr


# Usage example:
input_memmap = '/path/to/input_array.npy'
output_memmap = '/path/to/aligned_array.npy'

# Create input array (example shape: [1000, 512, 512, 3])
# np.save(input_memmap, your_large_array, allow_pickle=False)

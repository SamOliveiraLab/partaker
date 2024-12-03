# Model loading. TODO: move to another file

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras.optimizers import Adam  # Adam optimizer instead of SGD...
# from tensorflow.keras.optimizers.legacy import Adam  # Adam optimizer instead of SGD...
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Concatenate,
)

# Segmentation imports
from typing import Union, List, Tuple, Callable, Dict # Python types

#### Entropy and Loss functions ####
def pixelwise_weighted_binary_crossentropy_seg(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        print("Gone through an exception!");
        pass

    # Make background weights be equal to the model's prediction
    bool_bkgd = weight == 0 / 255
    weight = tf.where(bool_bkgd, y_pred, weight)

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=None,
    )

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)

    loss = tf.scalar_mul(
        10 ** 6, tf.scalar_mul(1 / tf.math.sqrt(tf.math.reduce_sum(weight)), loss)
    )

    return loss

############## U-NETS MODEL ##############
"""
A block of layers for 1 contracting level of the U-Net

Parameters
----------
input_layer : tf.Tensor
    The convolutional layer that is the output of the upper level's
    contracting block.
filters : int
    filters input for the Conv2D layers of the block.
conv2d_parameters : dict()
    kwargs for the Conv2D layers of the block.
dropout : float, optional
    Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
    layer is added.
    The default is 0
name : str, optional
    Name prefix for the layers in this block. The default is "Contracting".

Returns
-------
conv2 : tf.Tensor
    Output of this level's contracting block.

"""


# Contracting Block for the U-Net
def contracting_block(
    input_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Contracting",
) -> tf.Tensor:
    
    # Pooling layer: (sample 'images' down by factor 2)
    pool = MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
    
    # First Convolution layer
    conv1 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_1")(pool)
    
    # Second Convolution layer
    conv2 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_2")(conv1)
    
    # If a dropout is necessary, otherwise just return
    if (dropout == 0):
        return conv2;
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv2)
        return drop;

    
"""
A block of layers for 1 expanding level of the U-Net

Parameters
----------
input_layer : tf.Tensor
    The convolutional layer that is the output of the lower level's
    expanding block
skip_layer : tf.Tensor
    The convolutional layer that is the output of this level's
    contracting block
filters : int
    filters input for the Conv2D layers of the block.
conv2d_parameters : dict()
    kwargs for the Conv2D layers of the block.
dropout : float, optional
    Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
    layer is added.
    The default is 0
name : str, optional
    Name prefix for the layers in this block. The default is "Expanding".

Returns
-------
conv3 : tf.Tensor
    Output of this level's expanding block.

"""

# Expanding Block for the U-Net
def expanding_block(
    input_layer: tf.Tensor,
    skip_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Expanding",
) -> tf.Tensor:
    
    # Up-Sampling
    up = UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
    conv1 = Conv2D(filters, 2, **conv2d_parameters, name=name + "_Conv2D_1")(up)
    
    # Merge with skip connection layer
    merge = Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])
    
    # Convolution Layers
    conv2 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_2")(merge)
    conv3 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_3")(conv2)
    
    # If there needs dropout, otherwise, lets return
    if (dropout == 0):
        return conv3;
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv3)
        return drop;
    
"""
Unstacks the mask from the weights in the output tensor for
segmentation and computes binary accuracy

Parameters
----------
y_true : Tensor
Stack of groundtruth segmentation masks + weight maps.
y_pred : Tensor
Predicted segmentation masks.

Returns
-------
Tensor
Binary prediction accuracy.

"""
def unstack_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    try:
        print(y_true)
        print("y_true:", y_true.shape)
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    return keras.metrics.binary_accuracy(seg, y_pred)
    


# Actual U-net
"""
Generic U-Net declaration.

Parameters
----------
input_size : tuple of 3 ints, optional
    Dimensions of the input tensor, excluding batch size.
    The default is (256,32,1).
final_activation : string or function, optional
    Activation function for the final 2D convolutional layer. see
    keras.activations
    The default is 'sigmoid'.
output_classes : int, optional
    Number of output classes, ie dimensionality of the output space of the
    last 2D convolutional layer.
    The default is 1.
dropout : float, optional
    Dropout layer rate in the contracting & expanding blocks. Valid range
    is [0,1). If 0, no dropout layer is added.
    The default is 0.
levels : int, optional
    Number of levels of the U-Net, ie number of successive contraction then
    expansion blocks are combined together.
    The default is 5.

Returns
-------
model : Model
    Defined U-Net model (not compiled yet).

"""
def unet(
    input_size: Tuple[int, int, int] = (256, 32, 1),
    final_activation = "sigmoid",
    output_classes = 1,
    dropout: float = 0,
    levels: int = 5
) -> Model:
    
    # Default parameters for convolution
    conv2d_params = {
        "activation" : "relu",
        "padding" : "same",
        "kernel_initializer": "he_normal",
    }
    
    # Inputs Layer
    inputs = Input(input_size, name="true_input")
    
    # First level input convolutional layers:
    # We pass through 2 3x3 Convolution layers...
    filters = 64
    conv = Conv2D(filters, 3, **conv2d_params, name="Level0_Conv2D_1")(inputs)
    conv = Conv2D(filters, 3, **conv2d_params, name="Level0_Conv2D_2")(conv)
    
    # Generating Contracting Path (that is moving down the encoder block)
    level = 0;
    contracting_outputs = [conv];
    for level in range(1, levels):
        filters *= 2
        contracting_outputs.append(
            contracting_block(
                contracting_outputs[-1],
                filters,
                conv2d_params,
                dropout = dropout,
                name=f"Level{level}_Contracting",
            )
        )
    
    # Generating Expanding Path (that is moving up the decoder block)
    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = expanding_block(
            expanding_output,
            contracting_outputs.pop(),
            filters,
            conv2d_params,
            dropout = dropout,
            name=f"Level{level}_Expanding",
        )
    
    # Next we have the final output layer
    output = Conv2D(output_classes, 1, activation=final_activation, name="true_output")(expanding_output)
    model = Model(inputs=inputs, outputs=output)
    
    return model
    
# Unets Physical Model for Segmentation, think of it as a wrapper function...
def unet_segmentation(
    pretrained_weights = None,
    input_size: Tuple[int, int, int] = (256, 32, 1),
    levels: int = 5,
) -> Model: # Force a Model Class to come 
    
    # Run the following inputs into the unet algorithm defined above...
    model = unet(
        input_size = input_size,
        final_activation = "sigmoid",
        output_classes = 1,
        levels = levels,
    );
    
    # Learning rate 1e-4
    # loss = pixelwise_weighted_binary_crossentropy_seg,
    model.compile(
        optimizer = Adam(learning_rate = 1e-4),
        loss = pixelwise_weighted_binary_crossentropy_seg,
        metrics = [unstack_acc]
    )
    
    # If we have any pre-trained weights...
    if pretrained_weights:
        model.load_weights(pretrained_weights);
    
    return model

# target_size_seg = (512, 512)

# model = unet_segmentation(input_size = target_size_seg + (1,))
# # model.load_weights('./checkpoints/delta_2_29_01_24_5eps')
# # model.load_weights('./checkpoints/delta_2_19_02_24_200eps')
# model.load_weights('./checkpoints/delta_2_20_02_24_600eps')
# # model.summary()

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import imageio.v2 as imageio

from cachier import cachier
import datetime

# @cachier(stale_after=datetime.timedelta(days=3))
def segment_images(directory='/Users/hiram/Workspace/OliveiraLab/PartakerV3/src/aligned_data/XY8_Long_PHC', weights='/Users/hiram/Workspace/OliveiraLab/PartakerV3/src/checkpoints/delta_2_20_02_24_600eps.index'):

    target_size_seg = (512, 512)
    model = unet_segmentation(input_size = target_size_seg + (1,))
    test_images = Path(directory)
    imgs = list(map(lambda x : cv2.imread(str(x), cv2.IMREAD_GRAYSCALE), sorted([img for img in test_images.iterdir()], key=lambda x : int(x.stem))))

    def my_resize(img):
        a = img[55:960, 150:810]
        a = cv2.resize(a, (512, 512))
        a = np.expand_dims(a, axis=-1)
        return a

    pred_imgs = model.predict(np.array(list(map(my_resize, imgs))))
    return pred_imgs

# Attempt to use cellpose

from cellpose import models, io

# Initialize the Cellpose model
# model = models.Cellpose(model_type='cyto')
# cellposemodel = models.Cellpose(gpu=True, model_type='cyto')
# cellposemodel = models.Cellpose(gpu=True, model_type='cyto3')

import os

class CellposeModelSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CellposeModelSingleton, cls).__new__(cls, *args, **kwargs)
            
            # Check environment "PARTAKER_GPU": "1" or "0"
            if "PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1":
                cls._instance.model = models.CellposeModel(gpu=True, model_type='deepbacs_cp3')
            else:
                cls._instance.model = models.CellposeModel(gpu=False, model_type='deepbacs_cp3')

        return cls._instance

def preprocess_image(image):
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)

    # Apply Gaussian blur for denoising
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    return blurred_image

def segment_this_image(image):
    # Preprocess image before segmentation
    preprocessed_image = preprocess_image(image)

    # Use Cellpose for segmentation
    cellpose_inst = CellposeModelSingleton().model
    masks, flows, styles = cellpose_inst.eval(preprocessed_image, diameter=None, channels=[0, 0])

    # Create binary mask
    bw_image = np.zeros_like(masks, dtype=np.uint8)
    bw_image[masks > 0] = 255
    
    # Debug: Show binary mask
    # cv2.imshow("Binary Mask", bw_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return bw_image

def segment_all_images(images, progress=None):
    # Get the Cellpose model instance
    cellpose_inst = CellposeModelSingleton().model

    # Ensure images are in the correct format
    images = [img.squeeze() if img.ndim > 2 else img for img in images]

    # Run segmentation
    try:
        masks, _, _ = cellpose_inst.eval(images, diameter=None, channels=[0, 0])
        masks = np.array(masks)  # Ensure masks are a NumPy array
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None

    # Create binary black-and-white masks
    try:
        bw_images = np.zeros_like(masks, dtype=np.uint8)
        bw_images[masks > 0] = 255  # Convert labeled masks to binary
    except Exception as e:
        print(f"Error converting masks to binary: {e}")
        return None

    # Update progress if a callback is provided
    if progress:
        if callable(progress):  # If it's a function
            progress(len(images))
        else:  # Assume it's a PyQt signal
            progress.emit(len(images))

    return bw_images

"""
Segments one image and returns it, in a single channel
"""
def _segment_this_image(image):
    image = np.array(image)

    target_size_seg = (512, 512)
    model = unet_segmentation(input_size = target_size_seg + (1,))

    def my_resize(img):
        print(img.shape)
        # a = img[55:960][150:810]
        a = img
        a = cv2.resize(a, (512, 512))
        a = np.expand_dims(a, axis=-1)
        return a

    pred_imgs = model.predict(np.array([my_resize(image)]))

    print(pred_imgs.shape)
    return pred_imgs[0, :, :, 0]

def extract_individual_cells(image, segmented_image):
    """
    Extracts individual cells from the original image based on the segmented mask.

    Parameters:
    -----------
    image : np.ndarray
        The original grayscale or raw image.
    segmented_image : np.ndarray
        The binary segmented image where each cell is labeled uniquely.

    Returns:
    --------
    List of tuples where each tuple contains:
        - Cropped cell image (np.ndarray)
        - Bounding box (x, y, w, h)
    """
    # Ensure the images are the same size
    assert image.shape == segmented_image.shape, "Image and segmented image must have the same dimensions."

    # Find connected components in the segmented mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(segmented_image, connectivity=8)

    # Extract individual cells
    extracted_cells = []
    for label in range(1, num_labels):  # Skip the background (label=0)
        # Extract bounding box for the label
        x, y, w, h, area = stats[label]

        # Skip small regions (noise)
        if area < 50:
            continue

        # Crop the corresponding region from the original image
        cropped_cell = image[y:y + h, x:x + w]
        extracted_cells.append((cropped_cell, (x, y, w, h)))

    return extracted_cells


def extract_cells_and_metrics(image, segmented_image):
    """
    Extract individual cells, their bounding boxes, and metrics from a segmented image.

    Parameters:
    - image: np.ndarray, the original grayscale image.
    - segmented_image: np.ndarray, the binary segmented image.

    Returns:
    - cell_mapping: dict, a dictionary with cell IDs as keys and a dictionary of metrics and bounding boxes as values.
    """
    from skimage.measure import regionprops, label

    # Label connected regions in the segmented image
    labeled_image = label(segmented_image)

    # Extract properties for each labeled region
    cell_mapping = {}
    for region in regionprops(labeled_image, intensity_image=image):
        if region.area < 50:  # Filter out small regions (noise)
            continue

        # Calculate bounding box and metrics
        x1, y1, x2, y2 = region.bbox  # Bounding box coordinates
        metrics = {
            "area": region.area,
            "perimeter": region.perimeter,
            "equivalent_diameter": region.equivalent_diameter,
            "orientation": region.orientation,
        }

        # Add cell information to the mapping
        cell_id = len(cell_mapping) + 1
        cell_mapping[cell_id] = {
            "bbox": (x1, y1, x2, y2),
            "metrics": metrics,
        }

    return cell_mapping


def annotate_image(image, cell_mapping):
    """
    Annotate the original image with bounding boxes and IDs for detected cells.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input image is not a valid numpy array.")
    print(f"Annotating image of shape: {image.shape}")  # Debugging

    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Ensure it's in RGB format
    for cell_id, data in cell_mapping.items():
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, str(cell_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated


def annotate_binary_mask(segmented_image, cell_mapping):
    """
    Annotate the binary segmented mask with bounding boxes and cell IDs.

    Parameters:
    -----------
    segmented_image : np.ndarray
        The binary segmented mask (black and white).
    cell_mapping : dict
        Cell ID mapping with metrics and bounding boxes.

    Returns:
    --------
    annotated : np.ndarray
        Annotated binary mask with bounding boxes and labels.
    """
    # Ensure input is grayscale
    if len(segmented_image.shape) == 3:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to RGB for annotations
    annotated = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)

    for cell_id, data in cell_mapping.items():
        y1, x1, y2, x2 = data["bbox"]

        # Draw bounding box (e.g., green with 2px thickness)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add cell ID within the bounding box
        text_position = (x1 + 5, y1 + 15)  # Adjust for better placement
        cv2.putText(
            annotated,
            str(cell_id),  # Cell ID as string
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font scale
            (255, 255, 255),  # White text color
            1,  # Line thickness
            cv2.LINE_AA,  # Anti-aliased for smoother text
        )

    return annotated
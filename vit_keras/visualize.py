import cv2
import numpy as np
import tensorflow as tf
from . import layers, vit


def attention_map(model, image):
    """Get an attention map for an image and model using the technique
    described in Appendix D.7 in the paper (unofficial).

    Args:
        model: A ViT model
        image: An image for which we will compute the attention map.
    """
    size = model.input_shape[1]
    grid_size = int(np.sqrt(model.layers[5].output_shape[0][-2] - 1))

    # Prepare the input
    X = vit.preprocess_inputs(cv2.resize(image, (size, size)))[np.newaxis, :]  # type: ignore

    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if isinstance(l, layers.TransformerBlock)
    ]
    blocks = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    )
    num_layers = blocks.shape[0]
    num_heads = blocks.shape[2]
    reshaped = blocks.reshape(
        (num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), image.shape[:2])[..., np.newaxis]
    return (mask * image).astype("uint8")

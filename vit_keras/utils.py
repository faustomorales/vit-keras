import os
import typing
import warnings
from urllib import request
from http import client
import io
import pkg_resources
import validators
import numpy as np
import scipy as sp
import cv2

try:
    import PIL
    import PIL.Image
except ImportError:  # pragma: no cover
    PIL = None

ImageInputType = typing.Union[str, np.ndarray, "PIL.Image.Image", io.BytesIO]


def get_imagenet_classes() -> typing.List[str]:
    """Get the list of ImageNet 2012 classes."""
    filepath = pkg_resources.resource_filename("vit_keras", "imagenet2012.txt")
    with open(filepath) as f:
        classes = [l.strip() for l in f.readlines()]
    return classes


def read(filepath_or_buffer: ImageInputType, size, timeout=None):
    """Read a file into an image object
    Args:
        filepath_or_buffer: The path to the file or any object
            with a `read` method (such as `io.BytesIO`)
        size: The size to resize the image to.
        timeout: If filepath_or_buffer is a URL, the timeout to
            use for making the HTTP request.
    """
    if PIL is not None and isinstance(filepath_or_buffer, PIL.Image.Image):
        return np.array(filepath_or_buffer.convert("RGB"))
    if isinstance(filepath_or_buffer, (io.BytesIO, client.HTTPResponse)):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str) and validators.url(filepath_or_buffer):
        return read(request.urlopen(filepath_or_buffer, timeout=timeout), size=size)
    else:
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(
                "Could not find image at path: " + filepath_or_buffer
            )
        image = cv2.imread(filepath_or_buffer)
    if image is None:
        raise ValueError(f"An error occurred reading {filepath_or_buffer}.")
    # We use cvtColor here instead of just ret[..., ::-1]
    # in order to ensure that we provide a contiguous
    # array for later processing. Some hashers use ctypes
    # to pass the array and non-contiguous arrays can lead
    # to erroneous results.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size))


def load_weights_numpy(model, params_path, pretrained_top):
    """Load weights saved using Flax as a numpy array.

    Args:
        model: A Keras model to load the weights into.
        params_path: Filepath to a numpy archive.
        pretrained_top: Whether to load the top layer weights.
    """
    params_dict = np.load(params_path, allow_pickle=False)
    weights_dict = {}
    input_keys = list(params_dict.keys())
    pre_logits = any(l.name == "pre_logits" for l in model.layers)
    input_keys_taken = []
    output_keys_assigned = []
    output_weights = {w.name: w.numpy() for w in model.weights}
    if not pretrained_top:
        for key in ["head/kernel", "head/bias"]:
            if key in output_weights:
                del output_weights[f"{key}:0"]
            if key in input_keys:
                input_keys.remove(key)
    n_transformers = len(
        set(
            "/".join(k.split("/")[:2])
            for k in input_keys
            if k.startswith("Transformer/encoderblock_")
        )
    )
    n_transformers_out = len(
        set(
            "/".join(k.split("/")[:2])
            for k in output_weights
            if k.startswith("Transformer/encoderblock_")
        )
    )
    assert n_transformers == n_transformers_out, (
        f"Wrong number of transformers ("
        f"{n_transformers_out} in model vs. {n_transformers} in weights)."
    )

    def apply_weights(keyi, keyo):
        assert keyo in output_weights, f"{keyo} not in output weights."
        expected_shape = output_weights[keyo].shape
        actual_weights = params_dict[keyi]
        if keyi == "Transformer/posembed_input/pos_embedding":
            if expected_shape != actual_weights.shape:
                token, grid = actual_weights[0, :1], actual_weights[0, 1:]
                sin = int(np.sqrt(grid.shape[0]))
                sout = int(np.sqrt(expected_shape[1] - 1))
                warnings.warn(
                    "Resizing position embeddings from " f"{sin} to {sout}",
                    UserWarning,
                )
                zoom = (sout / sin, sout / sin, 1)
                grid = sp.ndimage.zoom(
                    grid.reshape(sin, sin, -1), zoom, order=1
                ).reshape(sout * sout, -1)
                actual_weights = np.concatenate([token, grid], axis=0)[np.newaxis]
        if "MultiHeadDotProductAttention" in keyi:
            actual_weights = actual_weights.reshape(expected_shape)
        actual_shape = actual_weights.shape
        assert expected_shape == actual_shape, (
            f"Shapes for layer: {keyi} / {keyo} do not match "
            f"({actual_shape} in weights vs. {expected_shape} in model."
        )
        weights_dict[keyo] = actual_weights
        input_keys_taken.append(keyi)
        output_keys_assigned.append(keyo)

    for tidx in range(n_transformers):
        for norm in ["LayerNorm_0", "LayerNorm_2"]:
            for inname, outname in [("scale", "gamma"), ("bias", "beta")]:
                apply_weights(
                    f"Transformer/encoderblock_{tidx}/{norm}/{inname}",
                    f"Transformer/encoderblock_{tidx}/{norm}/{outname}:0",
                )
        for mlpdense in [0, 1]:
            for subname in ["kernel", "bias"]:
                apply_weights(
                    f"Transformer/encoderblock_{tidx}/MlpBlock_3/Dense_{mlpdense}/{subname}",
                    f"Transformer/encoderblock_{tidx}/MlpBlock_3/Dense_{mlpdense}/{subname}:0",
                )
        for attvar in ["query", "key", "value", "out"]:
            for subname in ["kernel", "bias"]:
                apply_weights(
                    f"Transformer/encoderblock_{tidx}/MultiHeadDotProductAttention_1/{attvar}/{subname}",
                    f"Transformer/encoderblock_{tidx}/MultiHeadDotProductAttention_1/{attvar}/{subname}:0",
                )
    for attvar in ["embedding", "head", "pre_logits"]:
        if attvar == "head" and not pretrained_top:
            continue
        if attvar == "pre_logits" and not pre_logits:
            continue
        for subname in ["kernel", "bias"]:
            apply_weights(f"{attvar}/{subname}", f"{attvar}/{subname}:0")
    for attname in ["Transformer/posembed_input/pos_embedding"]:
        apply_weights(f"{attname}", f"{attname}:0")
    for iname, outname in [("cls", "class_token/cls:0")]:
        apply_weights(iname, outname)
    for inname, outname in [("scale", "gamma"), ("bias", "beta")]:
        apply_weights(
            f"Transformer/encoder_norm/{inname}",
            f"Transformer/encoder_norm/{outname}:0",
        )
    unused = [k for k in input_keys if k not in input_keys_taken]
    missing = [k for k in output_weights if k not in output_keys_assigned]
    if unused:
        warnings.warn(f"Did not use the following weights: {unused}", UserWarning)
    if missing:
        warnings.warn(f"Did not use the following weights: {missing}", UserWarning)
    weights = [weights_dict.get(k, output_weights[k]) for k in output_weights]
    model.set_weights(weights)

# vit-keras
This is a Keras implementation of the models described in [An Image is Worth 16x16 Words:
Transformes For Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf). It is based on an earlier implementation from [tuvovan](https://github.com/tuvovan/Vision_Transformer_Keras), modified to match the Flax implementation in the [official repository](https://github.com/google-research/vision_transformer).

The weights here are ported over from the weights provided in the official repository. See `utils.load_weights_numpy` to see how this is done (it's not pretty, but it does the job).

## Usage
Install this package using `pip install vit-keras`

You can use the model out-of-the-box with ImageNet 2012 classes using
something like the following.

```python
from vit_keras import vit, utils

image_size = 384
classes = utils.get_imagenet_classes()
model = vit.vit_b16(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)
url = 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg'
image = utils.read(url, image_size)
X = vit.preprocess_inputs(image).reshape(1, image_size, image_size, 3)
y = model.predict(X)
print(classes[y[0].argmax()]) # Granny smith
```

You can fine-tune using a model loaded as follows.

```python
image_size = 224
model = vit.vit_l32(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=200
)
# Train this model on your data as desired.
```

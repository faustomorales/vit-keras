import keras
import tensorflow as tf

from vit_keras import vit


def test_saving():
    inp = keras.layers.Input(shape=(256, 256, 3))
    base = vit.vit_b16(  # type: ignore
        image_size=256,
        pretrained=False,
        include_top=False,
        pretrained_top=False,
    )
    x = base(inp)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=x)
    opt = keras.optimizers.Adam()
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=["categorical_accuracy"])
    model.save("weights.h5")
    tf.saved_model.save(model, "weights")

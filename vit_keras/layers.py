# pylint: disable=arguments-differ,missing-function-docstring,missing-class-docstring,unexpected-keyword-arg,no-value-for-parameter,too-many-ancestors,abstract-method
try:
    import keras
except ImportError:
    from tensorflow import keras


@keras.utils.register_keras_serializable()
class ClassToken(keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        self.cls = self.add_weight(
            shape=(1, 1, self.hidden_size),
            initializer="zeros",
            trainable=True,
            name="cls",
            dtype="float32",
        )

    def call(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]
        cls_broadcasted = keras.ops.cast(
            keras.ops.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return keras.ops.concatenate([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable()
class AddPositionEmbs(keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding",
            dtype="float32",
        )

    def call(self, inputs):
        return inputs + keras.ops.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = keras.layers.Dense(hidden_size, name="query")
        self.key_dense = keras.layers.Dense(hidden_size, name="key")
        self.value_dense = keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = keras.layers.Dense(hidden_size, name="out")

    def attention(self, query, key, value):
        score = keras.ops.matmul(query, keras.ops.transpose(key, axes=[0, 1, 3, 2]))
        dim_key = keras.ops.cast(keras.ops.shape(key)[-1], score.dtype)
        scaled_score = score / keras.ops.sqrt(dim_key)
        weights = keras.ops.softmax(scaled_score, axis=-1)
        output = keras.ops.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = keras.ops.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return keras.ops.transpose(x, axes=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = keras.ops.transpose(attention, axes=[0, 2, 1, 3])
        concat_attention = keras.ops.reshape(
            attention, (batch_size, -1, self.hidden_size)
        )
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
@keras.utils.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = keras.Sequential(
            [
                keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}_Dense_0",
                ),
                keras.layers.Lambda(
                    lambda x: keras.activations.gelu(x, approximate=False)
                ),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(input_shape[-1], name=f"{self.name}_Dense_1"),
                keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = keras.layers.Dropout(self.dropout)

    def call(self, inputs, training=False):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

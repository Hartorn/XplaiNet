import copy

import numpy as np

from tensorflow.keras import Model, Input
from tensorflow.keras.activations import sigmoid

from tensorflow.keras.layers import (
    LocallyConnected1D,
    BatchNormalization,
    Activation,
    Reshape,
    Concatenate,
    Dense,
    Embedding,
    ZeroPadding1D
)
from tensorflow_addons.activations import mish
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead

DEFAULT_CONV_OPTS = {
    "padding": "valid",
    "kernel_size": 1,
    "strides": 1,
    "use_bias": False,
    "activation": None,
}


DEFAULT_DENSE_OPTS = {
    "use_bias": False,
    "activation": None,
}

def add_dense_block(
    in_layer,
    lconv_dim,
    prefix,
    activation,
    use_bn=True,
    activation_first_layer=None,
    options=None,
):
    if options is None:
        options = {}

    x_num_layer = in_layer
    for i, lconv_layer in enumerate(lconv_dim):
        name = f"{prefix}_{i}_"
        layer_opts = copy.deepcopy(DEFAULT_DENSE_OPTS)
        layer_opts.update(options)

        x_num_layer = Dense(
            lconv_layer, name=name + "dense", **layer_opts
        )(x_num_layer)
        if use_bn:
            x_num_layer = BatchNormalization(name=name + "nb")(x_num_layer)
        temp_activation = (
            activation_first_layer
            if i == 0 and activation_first_layer is not None
            else activation
        )
        x_num_layer = Activation(temp_activation, name=name + "activation")(x_num_layer)

    return x_num_layer


def add_local_conv_block(
    in_layer,
    lconv_dim,
    prefix,
    activation,
    use_bn=True,
    activation_first_layer=None,
    options=None,
):
    if options is None:
        options = {}

    x_num_layer = in_layer
    for i, lconv_layer in enumerate(lconv_dim):
        name = f"{prefix}_{i}_"
        layer_opts = copy.deepcopy(DEFAULT_CONV_OPTS)
        layer_opts.update(options)

        x_num_layer = LocallyConnected1D(
            filters=lconv_layer, name=name + "conv", **layer_opts
        )(x_num_layer)
        if use_bn:
            x_num_layer = BatchNormalization(name=name + "nb")(x_num_layer)
        temp_activation = (
            activation_first_layer
            if i == 0 and activation_first_layer is not None
            else activation
        )
        x_num_layer = Activation(temp_activation, name=name + "activation")(x_num_layer)

    return x_num_layer


def build_optimizer():
    return Lookahead(RectifiedAdam(1e-3), sync_period=6, slow_step_size=0.5)


def build_model(
    params,
    lconv_dim=[],
    lconv_num_dim=[],
    activation=None,
    optimizer=None,
    emb_size=16,
    activation_num_first_layer=None,
    output_activation="sigmoid",
    output_dim=1,
):
    if optimizer is None:
        optimizer = build_optimizer()
    if activation is None:
        activation = mish

    # Here, we get info necessary to build the model
    input_cat_dim = len(params["cat_cols"])
    input_bool_dim = len(params["bool_cols"])
    input_num_dim = len(params["num_cols"])
    # nb_channels = params["nb_channels"]

    # Inputs of the model
    inputs = []
    #  Layers to concat before output
    concats = []

    # Handling booleans
    if input_bool_dim > 0:
        input_bool_layer = Input(shape=(input_bool_dim,), name="input_bool")
        inputs.append(input_bool_layer)
        concats.append(input_bool_layer)

    # Handling numeric
    if input_num_dim > 0:
        input_num_layer = Input(shape=(input_num_dim,), name="input_num")  # *3
        inputs.append(input_num_layer)
        x_num_layer = input_num_layer

        if len(lconv_num_dim) != 0 and input_num_dim > 0:
            x_num_layer = Reshape((input_num_dim, 1), name="reshape_num_input")(
                x_num_layer
            )

        x_num_layer = add_local_conv_block(
            x_num_layer,
            lconv_num_dim,
            "block_num",
            activation,
            use_bn=False,
            activation_first_layer=activation_num_first_layer,  # "tanh",
            options=None,
        )

        nb_filters = lconv_num_dim[-1] if len(lconv_num_dim) > 0 else 1
        x_num_layer = Reshape((input_num_dim * nb_filters,), name="reshape_num_output")(
            x_num_layer
        )
        concats.append(x_num_layer)

    # Handling cat
    if input_cat_dim > 0:
        embeddings = []

        max_emb_size = int(min(np.log2(max(params["cat_modalities"]))+ 1, 50))

        for index in range(input_cat_dim):
            input_cat_layer = Input(shape=(1,), name=f"input_cat_{index}")
            inputs.append(input_cat_layer)
            emb_size = int(min(np.power(params["cat_modalities"][index], 1/2), 50))
            emb = Embedding(
                params["cat_modalities"][index] + 1,
                emb_size,
                name=f"large_emb_{index}",
            )(input_cat_layer)

            emb = Reshape((emb_size,), name=f"reshape_cat_input{index}")(emb)
            emb = add_dense_block(
                        emb,
                        lconv_dim,
                        f"block_cat_{index}",
                        activation,
                        use_bn=False,
                        activation_first_layer=None,
                        options=None,
            )
            nb_filters = lconv_dim[-1] if len(lconv_dim) > 0 else 1
            emb = Reshape((1, nb_filters), name=f"reshape_cat_input_{index}")(emb)
            # emb = Reshape((emb_size, nb_filters), name=f"reshape_cat_input{index}")(emb)
            # emb = ZeroPadding1D(padding=(0, max_emb_size-emb_size), name=f"padding_cat_input{index}")(emb)

            embeddings.append(emb)
        if len(embeddings)>1:
            concat_emb = Concatenate(axis=1)(embeddings)
        else:
            concat_emb = embeddings[0]
        print(concat_emb)
        # nb_filters = lconv_dim[-1] if len(lconv_dim) > 0 else 1
        # concat_emb = Reshape((input_cat_dim, max_emb_size), name="reshape_cat_input")(concat_emb)

    #     x_layer = add_local_conv_block(
    #         concat_emb,
    #         lconv_dim,
    #         "block_cat",
    #         activation,
    #         use_bn=False,
    #         activation_first_layer=None,
    #         options=None,
    #     )

    #     nb_filters = lconv_dim[-1] if len(lconv_dim) > 0 else 1
        nb_filters = lconv_dim[-1] if len(lconv_dim) > 0 else 1
        concat_emb = Reshape((input_cat_dim * nb_filters,), name="reshape_cat_output")(
            concat_emb
        )

        concats.append(concat_emb)

    if len(concats) > 1:
        concat = Concatenate()(concats)
    else:
        concat = concats[0]

    # For now, output is only for binary classification
    output = Dense(output_dim, activation=output_activation, name="output")(concat)

    model = Model(
        inputs=inputs,
        outputs=[output],
        name="explainable_model",
    )
    if output_dim == 2 or (output_dim == 1 and output_activation == "sigmoid"):
        loss = "binary_crossentropy"
    elif output_dim == 1:
        loss = "mse"
    else:
        loss = "sparse_categorical_crossentropy"

    model.compile(loss=loss, optimizer=optimizer)

    return model


def predict(model, input_model):

    log_reg_weights = model.get_layer("output").get_weights()[0]
    # log_reg_bias = model.get_layer("output").get_weights()[1][0]

    outputs = []
    shapes = []
    weights = []

    layers_names = [layer.name for layer in model.layers]
    for name in ["input_bool", "reshape_num_output", "reshape_cat_output"]:
        if name not in layers_names:
            continue
        layer = model.get_layer(name)
        outputs.append(layer.output)

    for klass in range(log_reg_weights.shape[1]):
        consumed = 0
        weights.append([])
        for name in ["input_bool", "reshape_num_output", "reshape_cat_output"]:
            if name not in layers_names:
                continue
            layer = model.get_layer(name)
            # outputs.append(layer.output)
            input_shape = layer.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            nb_channel = input_shape[-1] if len(input_shape) > 2 else 1
            nb_features = input_shape[-2] if len(input_shape) > 2 else input_shape[-1]
            nb_weights = nb_channel * nb_features
            weights[klass].append(
                log_reg_weights[:, klass][consumed : consumed + nb_weights].reshape(
                    nb_features, nb_channel
                )
            )
            shapes.append((nb_features, nb_channel))
            consumed += nb_weights
        

    explainable_model = Model(
        inputs=[model.input],
        outputs=[model.output, *outputs],
    )

    predictions = explainable_model.predict(input_model)
    probas = predictions[0]
    aggregated_explanation = []

    for klass in range(log_reg_weights.shape[1]):
        aggregated_explanation.append([])
        for weight_slice, shape_feat, raw_explanation in zip(
            weights[klass], shapes, predictions[1:]
        ):
            reshaped_expl = raw_explanation.reshape(-1, shape_feat[0], shape_feat[1])
            reshaped_weights = weight_slice.reshape(1, *weight_slice.shape)
            feature_explanation = (
                (reshaped_expl * reshaped_weights).sum(axis=-1).reshape(-1, shape_feat[0])
            )
            aggregated_explanation[-1].append(feature_explanation)
        aggregated_explanation[-1] = np.hstack(aggregated_explanation[-1])

    aggregated_explanation = np.stack(aggregated_explanation, axis=-1)

    results = np.zeros(aggregated_explanation.shape)
    for klass in range(log_reg_weights.shape[1]):
        print(aggregated_explanation.shape)

        for idx in range(aggregated_explanation.shape[1]):
            #expla_cpy = np.copy(aggregated_explanation)
            #expla_cpy[:, idx] = 0
            results[:, idx, klass] = aggregated_explanation[:, idx, klass].reshape(-1)
            # TODO take in account sigmoid, softmax, ... and bias
            # expla_cpy.sum(axis=-1)
            # results[:, idx] = probas.reshape(-1) - sigmoid(
            #     expla_cpy.sum(axis=-1) + log_reg_bias
            # ).numpy().reshape(-1)

    return probas, results


def encode(model, input_model):

    log_reg_weights = model.get_layer("output").get_weights()[0]
    # log_reg_bias = model.get_layer("output").get_weights()[1][0]

    outputs = []
    shapes = []
    weights = []

    layers_names = [layer.name for layer in model.layers]

    consumed = 0

    for name in ["input_bool", "reshape_num_output", "reshape_cat_output"]:
        if name not in layers_names:
            continue
        layer = model.get_layer(name)
        outputs.append(layer.output)
        input_shape = layer.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        nb_channel = input_shape[-1] if len(input_shape) > 2 else 1
        nb_features = input_shape[-2] if len(input_shape) > 2 else input_shape[-1]
        nb_weights = nb_channel * nb_features
        weights.append(
            log_reg_weights[consumed : consumed + nb_weights].reshape(
                nb_features, nb_channel
            )
        )
        shapes.append((nb_features, nb_channel))
        consumed += nb_weights

    explainable_model = Model(
        inputs=[model.input],
        outputs=[model.output, *outputs],
    )

    predictions = explainable_model.predict(input_model)
    probas = predictions[0]
    aggregated_explanation = []

    for shape_feat, raw_explanation in zip(shapes, predictions[1:]):
        aggregated_explanation.append(
            raw_explanation.reshape(-1, shape_feat[0] * shape_feat[1])
        )

    aggregated_explanation = np.hstack(aggregated_explanation)

    return probas, aggregated_explanation

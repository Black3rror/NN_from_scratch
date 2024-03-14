import argparse
import os

import numpy as np
import tensorflow as tf


def convert_model_to_c(model_path, templates_dir, save_dir, verbose=True):
    model = tf.keras.models.load_model(model_path)
    if verbose:
        model.summary()

    input_size = model.layers[0].input.shape[1]

    layers_info = []
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            raise ValueError("Only Dense layers are supported")
        if layer.activation.__name__ not in ["linear", "relu"]:
            raise ValueError("Only linear and relu activations are supported")

        layer_info = {}
        layer_info["n"] = layer.units
        layer_info["activation"] = layer.activation.__name__
        layer_info["weights"] = np.array(layer.get_weights()[0])    # shape: (input_size, n)
        layer_info["biases"] = np.array(layer.get_weights()[1])     # shape: (n,)

        layers_info.append(layer_info)

    with open(os.path.join(templates_dir, "model.h"), "r") as f:
        model_h = f.read()
    with open(os.path.join(templates_dir, "model.c"), "r") as f:
        model_c = f.read()

    model_h = model_h.replace("{input_size}", str(input_size))
    model_h = model_h.replace("{n_layers}", str(len(layers_info)))

    layers_size_h = ""
    layers_size_c = ""
    layer_weights = ""
    layer_biases = ""
    layers_weights = ""
    layers_biases = ""
    layers_activation = ""
    for i, layer_info in enumerate(layers_info):
        layers_size_h += "#define LAYER_{}_SIZE {}\n".format(i, layer_info["n"])
        layers_size_c += "LAYER_{}_SIZE, ".format(i)

        layer_weights += "float layer_{}_weights[]".format(i) + " = {" + ", ".join(map(str, layer_info["weights"].flatten())) + "};\n"
        layers_weights += "layer_{}_weights, ".format(i)

        layer_biases += "float layer_{}_biases[]".format(i) + " = {" + ", ".join(map(str, layer_info["biases"])) + "};\n"
        layers_biases += "layer_{}_biases, ".format(i)

        layers_activation += "{}, ".format(layer_info["activation"].upper())

    layers_size_c = layers_size_c[:-2]    # remove the last comma
    layers_weights = layers_weights[:-2]    # remove the last comma
    layers_biases = layers_biases[:-2]    # remove the last comma
    layers_activation = layers_activation[:-2]    # remove the last comma

    model_h = model_h.replace("{layers_size}", layers_size_h)
    model_c = model_c.replace("{layers_size}", layers_size_c)
    model_c = model_c.replace("{layer_weights}", layer_weights)
    model_c = model_c.replace("{layers_weights}", layers_weights)
    model_c = model_c.replace("{layer_biases}", layer_biases)
    model_c = model_c.replace("{layers_biases}", layers_biases)
    model_c = model_c.replace("{layers_activation}", layers_activation)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model.h"), "w") as f:
        f.write(model_h)
    with open(os.path.join(save_dir, "model.c"), "w") as f:
        f.write(model_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--templates_dir", type=str, default="nn_from_scratch/model/c_templates", help="Path to the directory with the templates")
    parser.add_argument("--save_dir", type=str, required=False, help="Path to the directory to save the converted model")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = "c_files"

    convert_model_to_c(args.model_path, args.templates_dir, args.save_dir)

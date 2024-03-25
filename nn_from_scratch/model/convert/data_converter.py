import argparse
import os

import numpy as np
import tensorflow as tf


def convert_data_to_c(data_x, data_y, templates_dir, save_dir):
    """
    Convert the data to C format and save it to the specified directory.

    Args:
        data_x (np.ndarray): Input data.
        data_y (np.ndarray): Output data.
        templates_dir (str): Path to the directory with the templates.
        save_dir (str): Path to the directory to save the converted data.
    """
    with open(os.path.join(templates_dir, "data.h"), "r") as f:
        data_h = f.read()
    with open(os.path.join(templates_dir, "data.c"), "r") as f:
        data_c = f.read()

    assert data_x.shape[0] == data_y.shape[0], "The number of samples in data_x and data_y should be equal"
    assert data_x.ndim == 2, "data_x should be a 2D array"
    assert data_y.ndim == 2, "data_y should be a 2D array"

    data_h = data_h.replace("{n_samples}", str(data_x.shape[0]))
    data_h = data_h.replace("{input_size}", str(data_x.shape[1]))
    data_h = data_h.replace("{output_size}", str(data_y.shape[1]))
    data_c = data_c.replace("{input_size}", str(data_x.shape[1]))
    data_c = data_c.replace("{output_size}", str(data_y.shape[1]))

    samples_x = "\n"
    samples_y = "\n"
    for i in range(data_x.shape[0]):
        samples_x += "    {" + ", ".join(map(str, data_x[i])) + "},\n"
        samples_y += "    {" + ", ".join(map(str, data_y[i])) + "},\n"

    data_c = data_c.replace("{samples_x}", samples_x)
    data_c = data_c.replace("{samples_y}", samples_y)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "data.h"), "w") as f:
        f.write(data_h)
    with open(os.path.join(save_dir, "data.c"), "w") as f:
        f.write(data_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--templates_dir", type=str, default="nn_from_scratch/model/c_templates", help="Path to the directory with the templates")
    parser.add_argument("--save_dir", type=str, default="c_files", help="Path to the directory to save the converted model")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    input_size = model.layers[0].input_shape[1]

    random_data_x = np.random.rand(10, input_size).astype(np.float32)
    random_data_y = model.predict(random_data_x)

    convert_data_to_c(random_data_x, random_data_y, args.templates_dir, args.save_dir)

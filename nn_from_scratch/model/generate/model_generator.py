import datetime
import importlib
import logging
import os

import hydra
import numpy as np
import tensorflow as tf
import wandb
import yaml
from omegaconf import OmegaConf

from nn_from_scratch.model.convert.data_converter import convert_data_to_c
from nn_from_scratch.model.convert.model_converter import convert_model_to_c
from nn_from_scratch.model.generate.model import create_model, train_model, get_params_count, get_FLOPs, save_model, save_weights, log_model_to_wandb, measure_execution_time
from nn_from_scratch.model.generate.utils import get_abs_path


logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"
config_file_path = "nn_from_scratch/model/generate/configs/model_generator_config.yaml"
config_file_path = get_abs_path(config_file_path)

# default configs
denses_params = [16]             # each element is the number of neurons of a dense layer
activation = "relu"
epochs = 10
batch_size = 32
dataset_info = {
    "name": "sinus",
    "path": "nn_from_scratch/model/generate/data/sinus/data.py",
    "args": {
        "n_samples": 1000,
        "test_ratio": 0.2,
        "random_seed": 42,
    }
}
random_seed = 42
learning_rate = 1e-3


def set_model_configs_from_file(path):
    """
    Set the model configs from the config file.

    Args:
        path (str): Path to the config file.
    """
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    global denses_params, activation, epochs, batch_size, dataset_info, random_seed, learning_rate

    if "denses_params" in cfg:
        denses_params = cfg["denses_params"]
    if "activation" in cfg:
        activation = cfg["activation"]
    if "epochs" in cfg:
        epochs = cfg["epochs"]
    if "batch_size" in cfg:
        batch_size = cfg["batch_size"]
    if "dataset" in cfg:
        dataset_info = cfg["dataset"]
    if "random_seed" in cfg:
        random_seed = cfg["random_seed"]
    if "learning_rate" in cfg:
        learning_rate = cfg["learning_rate"]


def save_model_info(model_info, save_dir):
    """
    Saves the model info to the specified directory.

    Args:
        model_info (dict): The model info.
        save_dir (str): The directory where the model info should be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    yaml.Dumper.ignore_aliases = lambda *args : True
    with open(os.path.join(save_dir, "model_info.yaml"), 'w') as f:
        yaml.dump(model_info, f, indent=4, sort_keys=False)


@hydra.main(config_path=os.path.dirname(config_file_path),
            config_name=os.path.splitext(os.path.basename(config_file_path))[0],
            version_base=None)
def generate_models(cfg):

    for target in cfg.targets:
        cfg.target_buf = target
        cfg.time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        wandb_name = datetime.datetime.strptime(cfg.time_tag, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
        wandb_dir = get_abs_path(cfg.model_save_dir)
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(project=cfg.wandb_project_name, tags=[target], name=wandb_name, dir=wandb_dir)

        title = "Creating the model with setting {}".format(target)
        print("\n")
        print("="*80)
        print("-"*((80-len(title)-2)//2), end=" ")
        print(title, end=" ")
        print("-"*((80-len(title)-2)//2))
        print("="*80)

        # update configs from the config file
        set_model_configs_from_file(cfg.model_config_path)

        # load corresponding dataset
        spec = importlib.util.spec_from_file_location("imported_module", dataset_info["path"])
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)

        # load data and create model
        dataset = imported_module.DatasetSupervisor(**dataset_info["args"])

        input_size = dataset.feature_size
        output_size = dataset.num_labels
        output_activation = dataset.output_activation
        model = create_model(input_size, denses_params, output_size, activation, output_activation, random_seed)

        # compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss=dataset.loss_function, metrics=dataset.metrics)

        # model info
        print("Model summary:")
        model.summary()
        print("")
        total_params, trainable_params, non_trainable_params = get_params_count(model)
        MACs = get_FLOPs(model) // 2

        # train the model
        print("Training the model ...")
        tensorboard_log_dir = os.path.join(cfg.model_save_dir, 'tf/logs')
        best_weights_dir = os.path.join(cfg.model_save_dir, 'tf/weights/weights_best')
        train_model(model, dataset.train_x, dataset.train_y, epochs, batch_size, dataset.test_x, dataset.test_y, tensorboard_log_dir, best_weights_dir, True, random_seed)
        print("")

        # evaluate the model
        evaluation_result = None
        if cfg.evaluate_models:
            try:
                eval_vals = model.evaluate(dataset.test_x, dataset.test_y, verbose=0)
                print(dataset.loss_function, ":", eval_vals[0])
                for i in range(len(dataset.metrics)):
                    print(dataset.metrics[i], ":", eval_vals[i+1])
                print("")
            except Exception as e:
                print("Error in evaluating the model: {}".format(e))
                print("Continuing without evaluation")
                continue

        # save the model and weights
        print("Saving model and weights to the directory: {} ...".format(cfg.model_save_dir), end=" ", flush=True)
        save_model(model, os.path.join(cfg.model_save_dir, "tf/model"))
        save_weights(model, os.path.join(cfg.model_save_dir, 'tf/weights/weights_last'))
        print("Done\n")
        log_model_to_wandb(os.path.join(cfg.model_save_dir, "tf/model"), "{}".format(target.replace("/", "_")))

        # convert the model and data to C
        print("Converting the model to C ...", end=" ", flush=True)
        convert_model_to_c(os.path.join(cfg.model_save_dir, "tf/model/keras_format/model.keras"), cfg.c_templates_dir, cfg.c_save_dir, verbose=False)
        print("Done\n")

        print("Converting the data to C ...", end=" ", flush=True)
        eq_data_x = dataset.train_x[:cfg.n_eqcheck_data]
        eq_data_y = model.predict(eq_data_x, verbose=0)
        convert_data_to_c(eq_data_x, eq_data_y, cfg.c_templates_dir, cfg.c_save_dir, file_name="eqcheck_data")
        print("Done\n")

        # measure the execution time
        if cfg.measure_execution_time:
            print("Measuring execution time ...")
            x = np.array([dataset.train_x[0]])
            execution_time = measure_execution_time(model, x)
            print("Average run time: {} ms\n".format(execution_time))

        # save the model info
        model_info = {"Description": ""}
        model_info["denses_params"] = denses_params
        model_info["activation"] = activation
        model_info["epochs"] = epochs
        model_info["batch_size"] = batch_size
        model_info["learning_rate"] = learning_rate
        model_info["dataset"] = dataset_info
        model_info["random_seed"] = random_seed
        model_info["total_params"] = total_params
        model_info["trainable_params"] = trainable_params
        model_info["non_trainable_params"] = non_trainable_params
        model_info["MACs"] = MACs
        if evaluation_result is not None:
            for metric, value in evaluation_result.items():
                if not isinstance(metric, str):
                    metric = str(metric)
                model_info[metric] = value
        if cfg.measure_execution_time:
            model_info["execution_time"] = execution_time
        model_info["wandb_name"] = wandb_name

        print("Saving the model info in the directory: {} ...".format(cfg.model_save_dir), end=" ", flush=True)
        save_model_info(model_info, cfg.model_save_dir)
        print("Done\n")
        wandb.config.update(model_info)

        wandb.finish()

    print("Done\n")


if __name__ == "__main__":
    generate_models()

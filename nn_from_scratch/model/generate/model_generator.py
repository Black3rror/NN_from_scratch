import datetime
import logging
import os

import hydra
import wandb

from nn_from_scratch.model.convert.model_converter import convert_model_to_c
from nn_from_scratch.model.convert.data_converter import convert_data_to_c
from nn_from_scratch.model.generate.model import ModelSupervisor
from nn_from_scratch.model.generate.utils import get_abs_path


logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"
config_file_path = "nn_from_scratch/model/generate/configs/model_generator_config.yaml"
config_file_path = get_abs_path(config_file_path)


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

        supervisor = ModelSupervisor(cfg.model_config_path)

        try:
            supervisor.compile_model(fine_tuning=False)
        except Exception as e:
            print("Error in compiling the model: {}".format(e))
            print("Continuing without compilation")
            continue

        print("Model summary:")
        supervisor.model.summary()
        print("")
        total_params, trainable_params, non_trainable_params = supervisor.get_params_count()
        MACs = supervisor.get_FLOPs() // 2

        if cfg.train_models:
            print("Training the model ...")
            tensorboard_log_dir = os.path.join(cfg.model_save_dir, 'tf/logs')
            best_weights_dir = os.path.join(cfg.model_save_dir, 'tf/weights/weights_best')
            supervisor.train_model(fine_tuning=False, tensorboard_log_dir=tensorboard_log_dir, best_weights_dir=best_weights_dir, use_wandb=True)
            print("")

        evaluation_result = None
        if cfg.evaluate_models:
            try:
                evaluation_result = supervisor.evaluate_model()
                for metric, value in evaluation_result.items():
                    print(metric, ":", value)
                print("")
            except Exception as e:
                print("Error in evaluating the model: {}".format(e))
                print("Continuing without evaluation")
                continue

        print("Saving model and weights to the directory: {} ...".format(cfg.model_save_dir), end=" ", flush=True)
        supervisor.save_model(os.path.join(cfg.model_save_dir, "tf/model"))
        supervisor.save_weights(os.path.join(cfg.model_save_dir, 'tf/weights/weights_last'))
        print("Done\n")
        supervisor.log_model_to_wandb(os.path.join(cfg.model_save_dir, "tf/model"), "{}".format(target.replace("/", "_")))

        print("Converting the model to C ...", end=" ", flush=True)
        convert_model_to_c(os.path.join(cfg.model_save_dir, "tf/model/keras_format/model.keras"), cfg.c_templates_dir, cfg.c_save_dir, verbose=False)
        print("Done\n")

        print("Converting the data to C ...", end=" ", flush=True)
        eq_data_x, eq_data_y = supervisor.get_eqcheck_data(cfg.n_eqcheck_data)
        convert_data_to_c(eq_data_x, eq_data_y, cfg.c_templates_dir, cfg.c_save_dir)
        print("Done\n")

        if cfg.measure_execution_time:
            print("Measuring execution time ...")
            execution_time = supervisor.measure_execution_time()
            print("Average run time: {} ms\n".format(execution_time))

        model_info = {"Description": ""}
        model_info["trained"] = cfg.train_models
        model_info.update(supervisor.get_model_info())
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
        supervisor.save_model_info(model_info, cfg.model_save_dir)
        print("Done\n")
        wandb.config.update(model_info)

        wandb.finish()

    print("Done\n")


if __name__ == "__main__":
    generate_models()

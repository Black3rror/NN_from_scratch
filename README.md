# nn_from_scratch

The main objective is to write a C code that runs a fully connected model on microcontrollers and is capable of training it on the device as well.

This project consists of two parts:
- *nn_from_scratch/model*: Generate and train a fully connected model in TensorFlow. Further, convert its information and some data samples to C code.
- *nn_from_scratch/hardware*: Use the generated C code to run/train the model on a microcontroller.

## How to use

1. Clone the project
2. Setup the project
    - If using *Makefile*, run `make setup_project`
    - Otherwise, or in case of any issues, manually execute the commands in the *setup_project* target of the *Makefile*
3. Generate the C code
    1. Not having a TensorFlow model: You can use this project to generate a model and convert it to C code. For this, you need to run `make generate_models` or `python -m nn_from_scratch.model.generate.model_generator`
        - *nn_from_scratch\model\generate\configs\model_generator_config.yaml* contains general settings that you can change. Mainly, it contains a list of models to generate.
        - For each model listed in the `targets` variable of the *model_generator_config.yaml* file, there should be a corresponding YAML file in *nn_from_scratch\model\generate\configs* that describes your model (like *setting_1.yaml*). Create them, or change them as needed.
    2. Already having a TensorFlow model: You can convert it to C code by running `python -m nn_from_scratch.model.convert.model_converter --model_path <path_to_model>`
        - Run `python -m nn_from_scratch.model.convert.model_converter --help` for more information.
4. Run the model on a microcontroller
    1. To be completed ...

## Project structure

This section is out of date and reqires updating once the project structure is finalized.

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make setup_project` or `make requirements`
├── README.md            <- The top-level README for developers using this project.
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml       <- Project configuration file
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
|
├── requirements_test.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── nn_from_scratch  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── get_data.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── train_model.py   <- script for training the model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [DL_project_template](https://github.com/Black3rror/DL_project_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for
starting a Deep Learning Project.

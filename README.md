## Environments
First, you should setup your virtual environment. The instructions for setup using Conda and Pip are detailed below. 

**Note**: If you are having trouble rendering interactive plotly figures and you're using the pip + virtualenv method, try using Conda instead.

### Conda 

Conda uses the provided `environment.yml` file.
You can ignore `requirements.txt` if you choose this method.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate ift6758-conda-env

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-conda-env

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment file using the `create` command:

    conda env export > environment.yml

### Pip + Virtualenv

An alternative to Conda is to use pip and virtualenv to manage your environments.
This may play less nicely with Windows, but works fine on Unix devices.
This method makes use of the `requirements.txt` file; you can disregard the `environment.yml` file if you choose this method.

Ensure you have installed the [virtualenv tool](https://virtualenv.pypa.io/en/latest/installation.html) on your system.
Once installed, create a new virtual environment:

    vitualenv ~/ift6758-venv
    source ~/ift6758-venv/bin/activate

Install the packages from a requirements.txt file:

    pip install -r requirements.txt

As before, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-venv

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you want to create a new `requirements.txt` file, you can use `pip freeze`:

    pip freeze > requirements.txt


## Installation
Once you've setup your environment, you can install this package by running the following command from the root directory of your repository. 

    pip install -e .

You should see something similar to the following output:

    > pip install -e .
    Obtaining file:///home/USER/project-template
    Installing collected packages: ift6758
    Running setup.py develop for ift6758
    Successfully installed ift6758-0.1.0


## How To Run
Before everything, the first 2 things you should run are the `data_cleaning_demo.ipynb` and `get_train_and_test_sets_demo.ipynb` notebooks, which can be found under the `notebooks` folder, and in this order specifically. These may take a while to run, but they will give all the data needed to run the rest of the experiments and cache it, so access will be much faster.

### Demos and visualizations
To see the rest of our experiments, you are free to run the rest of the notebooks found in the `notebooks` folder, as well as the files and notebooks in the following sections:

- `features`: All our feature engineering experiments
- `models`: The models we built and used for goal prediction
- `visualizations`: Our visual explorations of the data. **NOTE:** the `rink_plot_widget` file is not supposed to be executed, and is used in the game_event_explorer.ipynb notebook. Also, the `hockey_visualization_app.py` file has to be run using the command

        streamlit run hockey_visualization_app.py


### Goal Prediction Application (With Docker)
You can use the Goal Prediction Streamlit application from your machine using Docker Desktop by following these steps:

1. Start Docker Desktop
2. Open a terminal to the root of the project on your machine
3. Run the command

    docker compose build
4. Run the command

    docker compose up

The Streamlit app should now be running on your localhost. To close everything, run the command

    docker compose down

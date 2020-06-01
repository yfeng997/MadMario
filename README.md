# MadMario
Interactive tutorial to build a learning Mario, for first-time RL learners

## Set Up
1. Install [conda](https://www.anaconda.com/products/individual)
2. Install dependencies with `environment.yml`
    ```
    conda env create -f environment.yml
    ```
    Check the new environment *mario* is [created successfully](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

3. Activate *mario* enviroment
    ```
    conda activate myenv
    ```

## Project Structure
**main.py**
Loop for action, memorizing and learning. To start the agent:
```
python main.py
```

**tutorial.ipynb**
Interactive tutorial with extensive explanation and feedback. Run it on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).

**agent.py**
Define how the agent collects experiences, makes actions given observations and updates the action policy.

**wrappers.py**
Environment pre-processing logics, including observation resizing, rgb to grayscale, etc.

**neural.py**
Define a small convolution neural network as the Q value estimator.

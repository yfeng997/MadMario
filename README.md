
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

## Running
To start the **learning** process for Mario, 
```
python main.py
```
This will by default start the *double Q-learning* training. Logs are printed to the terminal and outputted to a `log.txt` under `checkpoints/curr_date_time/`. Every once in a while, a couple of trained Marios will be stored under the same folder named `online_q_idx.chkpt`. We save multiple Marios at once because RL is [known to be unstable](alexirpan.com/2018/02/14/rl-hard.html), and we need a couple examples to filter out the noise. 

Depending on if a CUDA device is available, training process will automatically happen on GPU/CPU. Estimated training time is 40 hours for CPU and 10 hours for GPU. 

To **test** a trained Mario, 
```
python replay.py
```
This will by default look at the most recent  folder under `checkpoints/`, e.g. `checkpoints/2020-06-06T22-00-00`, and sweep through all the saved Marios there. Change the `load_dir` parameter in `agent.replay()` if you wanna look at agents from a specific timestamp. 


## Project Structure
**main.py**
Loop for action, memorizing and learning.

**tutorial.ipynb**
Interactive tutorial with extensive explanation and feedback. Run it on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).  

**agent.py**
Define how the agent collects experiences, makes actions given observations and updates the action policy.

**wrappers.py**
Environment pre-processing logics, including observation resizing, rgb to grayscale, etc.

**neural.py**
Define a small convolution neural network as the Q value estimator.


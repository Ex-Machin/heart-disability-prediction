This is my diploma. Here I tried to predict hear reat disability, using Tensorflow neural network.

Database used: https://www.physionet.org/content/ptb-xl/1.0.3/


## Create virtual environment

### Use a version of Python that is less than 3.10
conda create --name your_env_name python<3.10

### Activate new environment
conda activate your_env_name

### Install ipykernel
conda install -c anaconda ipykernel

### Add this new environment to your Jupyter Notebook kernel list
ipython kernel install --name your_env_name --user

Windows only: When trying to launch `jupyter notebook`, you may receive a win32api error.
The command below fixes that issue.

conda install -c anaconda pywin32

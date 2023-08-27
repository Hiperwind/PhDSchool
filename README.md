# HIPERWIND PhD 2023 Summer School

## Cloning the repository
You can simply clone the repo via git: 
```
git clone https://github.com/Hiperwind/PhDSchool.git
```

## Installation 
We recommend the installation of a Python 3.10 virtual environment. You can do so via:
* Conda (recommended)
```
conda create -n "hiperwind" python=3.10
```
* venv (this option requires a prior Python installation)
```
python -m venv /path_to_new_virtual_environment
```
You can then pip install the required packages:
```
pip install -r requirements.txt
```

## Activate the virtual environment 
* Conda (recommended)
```
conda activate hiperwind
```
* venv 
```
source /path_to_new_virtual_environment/bin/activate
```

## Slides

The slides are available [here](/slides/).

## Exercises and tutorials
The exercises and tutorials are available [here](/exercises/).

## Project
A template containing all instructions necessary for the project is available [here](/project/). You can complete the project at your own pace, though we suggest the following timeframe:
* Day 1: Intro, installation of the Python environment, data visualization.
* Day 2: Environmental data fitting.
* Day 3: Surrogate model construction and testing.
* Day 4: Limit state definition and reliability estimation.

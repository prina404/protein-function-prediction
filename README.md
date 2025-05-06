# Protein function prediction
Project for the Deep Learning for Life Sciences 4EU+ hackathon

## Prerequisites

The custom `Dataset` classes in this project require that the **ProtVec** dataset [[link](http://dx.doi.org/10.7910/DVN/JMFHTN)] has been extracted into the `data` folder.

## Installation 

- Clone this repo:
    ```bash
    $ git clone https://github.com/prina404/protein-function-prediction.git 
    $ cd protein-function-prediction/
    ```

- Initialize a python virtual environment and activate it:
    ```bash
    $  python3 -m venv .venv && source .venv/bin/activate
    ```
- Install this project as a package
    ```bash
    $ pip install -e .
    ```
Now any `*.py` module under the `src/` folder can be imported, and `src/` is automatically set as its root folder. 
For example:

```python
# this can be a script located anywhere in this repo
import ProteinDataset   # this imports src/ProteinDataset.py
import utils.config     # this imports src/utils/config.py

...
```
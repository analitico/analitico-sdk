
## Analitico SDK

This package contains plugins and classes used to access analitico.ai cloud services and machine learning models. The package can be installed in Jupyter notebooks, Colaboratory notebooks or other Python environments. To access assets stored in Analitico you will need an API token.

### Installation

To install in Python:  
`pip install analitico`

To install on Jupyter, Colaboratory, etc:  
`!pip install analitico`

### Usage

```python
import analitico
import pandas as pd

# authorize calls with developer token
sdk = analitico.authorize_sdk(token="tok_xxx")

# download a dataset from analitico
dataset = sdk.get_dataset("ds_xxx")

# convert dataset to a typed pandas dataframe
df = dataset.get_dataframe()
```

### Testing

To run tests:  
`python -m pytest`

### Documenting code

Please use docstrings, see:  
https://www.datacamp.com/community/tutorials/docstrings-python

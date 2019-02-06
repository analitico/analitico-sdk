
# Analitico SDK

This library contains plugins and classes used to access analitico.ai cloud services and machine learning models.  

The library can be easily installed in Jupyter notebooks or other Python environments.

To install in Python:  
`pip install analitico`

To install on Jupyter, Colaboratory, etc:  
`!pip install analitico`

Usage example:  
```python
import analitico
import pandas as pd

api = analitico.authorize("tok_xxx")
dataset = api.get_dataset("ds_xxx")
df = dataset.get_dataframe()
```

## Packaging analitico

Python packaging user guide:  
https://packaging.python.org/

Prerequisites:  
`python3 -m pip install --user --upgrade setuptools wheel`

Compile packages:  
`python3 setup.py sdist bdist_wheel`

Publish to test.pypi.org:  
`python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*`

Publish to pypi.org:  
`python3 -m twine upload dist/*`

Package is visible at:  
`https://pypi.org/project/analitico/`

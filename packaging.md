## Packaging analitico

Python packaging user guide:  
https://packaging.python.org/

Prerequisites:  
```console
python3 -m pip install --user --upgrade setuptools wheel
pip install twine
pip install wheel
```

Compile packages:  
```console
rm -r dist
python3 source/setup.py sdist bdist_wheel
```

Publish to test.pypi.org:  
```console
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Publish to pypi.org:  
```console
python3 -m twine upload dist/*
```

Package is visible at:  
https://pypi.org/project/analitico/

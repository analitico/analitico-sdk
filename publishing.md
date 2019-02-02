
Python packaging user guide:  
https://packaging.python.org/

Compile packages:  
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

Publish to test.pypi.org:  
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

Publish to pypi.org:  
python3 -m twine upload dist/*

Package is visible at:  
https://pypi.org/project/analitico/

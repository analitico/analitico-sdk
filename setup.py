import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="analitico",
    version="2019.07.31.dev",
    author="Analitico Labs Inc.",
    author_email="info@analitico.ai",
    description="A Python package for Analitico.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/analitico/analitico-sdk",
    packages=setuptools.find_packages(),
    package_data={
        "analitico": ["test/assets/*.*"]
        },
    install_requires=[
        "catboost==0.11.1", 
        "numpy>=1.14.6", 
        "pandas>=0.22.0",
        "simplejson==3.16.0"
        ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)

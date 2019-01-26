# ML Deploy
This package provides the tooling to trial machine learning models in a 
pseudo-production environment. It provides the following utilities:
* Stores all models and model versions as serialized objects
* Stores all raw & processed data used for training & predicting
* Snapshots result data
* Provides model performance metrics in the form of data sets that are designed for easy data visualization.

This library is designed to be used with binary classification models.

ML Deploy has a dedicated data model for capturing your data, and can operate on any
database system supported by SQLAlchemy.

## Setup
1. Copy the ml_deploy directory to your local machine. This package is not currently registered on PyPI, so you'll need to have the package files.
2. From the command line, navigate to the ml_deploy package directory
    `$/ml_deploy>`
3. Install using pip
    `$/ml_deploy>pip install .`
		OR
		`$/ml_deploy>pip install -e .` to create a symlink that will capture code changes to the library.
4. Verify package using tests (optional)
    `$/ml_deploy>python setup.py test`

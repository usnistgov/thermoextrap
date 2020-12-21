# test/coverage
pytest --cov=cmomy --cov-report html -x -v

# convert readme.org to readme.md
pandoc README.org -o README.md

# create distnoted
python setup.py sdist bdist_wheel

# upload to pypi
twine upload dist/*

# create conda dist from pypi
mkdir conda_dist
grayskull pypi cmomy
conda-build cmomy
anaconda upload /path/to/cmomy


# create conda dist from git
cd recipe/github/
conda-build .


# run tests on pip/conda
conda create -n cmomy_pip python=3.8 pytest
conda activate cmomy_pip
pip install cmomy
pytest -x -v ...

conda create -n cmomy_conda python=3.8 pytest
conda activate cmomy_conda
conda install -c wpk-nist cmomy
pytest -x -v ...

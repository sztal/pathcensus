.PHONY: help clean clean-pyc clean-build clean-test lint test test-all coverage docs release sdist

help:
	@echo "clean - remove auxiliary files/artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove testing artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - compute code coverage quickly with the default Python"
	@echo "cov-report - display coverage report"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "sdist - package"

clean: clean-build clean-py clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.eggs' -exec rm -rf {} +

clean-py:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.nbc' -exec rm -f {} +
	find . -name '*.nbi' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-test:
	find . -name '.benchmarks' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '.tox' -exec rm -rf {} +

lint:
	pytest --pylint -m pylint

test:
	pytest

test-all:
	tox

test-all-p:
	tox -p auto

cov-run:
	coverage run --source pathcensus setup.py test

cov-report:
	coverage report --omit=pathcensus/core/*.py
	coverage html --omit=pathcensus/core/*.py
	xdg-open htmlcov/index.html || open htmlcov/index.html

coverage: cov-run cov-report

codecov-xml:
	pytest --cov=./pathcensus --cov-report=xml

docs:
	rm -f docs/pathcensus.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pathcensus
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	xdg-open docs/_build/html/index.html || open docs/_build/html/index.html

release: clean
	python setup.py sdist upload
	python setup.py bdist_wheel upload

sdist: clean
	python setup.py sdist
	python setup.py bdist_wheel upload
	ls -l dist

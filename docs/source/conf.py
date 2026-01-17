# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
import os
#sys.path.insert(0, os.path.abspath('../../Improve-Deep-NN/NN/src'))


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

NN_SRC = os.path.join(
    PROJECT_ROOT,
    "Improve-Deep-NN",
    "NN",
    "src",
)

sys.path.insert(0, NN_SRC)

project = 'Deep Learning From Scratch'
copyright = '2026, Nirav'
author = 'Nirav'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon"
        ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# for deployment

html_baseurl = "https://USERNAME.github.io/REPO/"

extensions += ["sphinx.ext.githubpages"]

# moch the import, there is no real need to import numpy 
autodoc_mock_imports = [
    "numpy",
]

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
import os

# Define paths
project_root = Path(__file__).parent
module_path = Path(project_root, "src")

# Insert paths to sys.path
sys.path.insert(0, str(module_path.resolve()))
sys.path.insert(0, str(project_root.resolve()))

project = 'Geti Instant Learn'
copyright = 'Intel Corporation'
author = 'Intel Corporation'
release = os.environ.get('VERSION', '1.0')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # this is the plugin to generate api docs
    'sphinx_autodoc_typehints',  # another part of this plugin
    'sphinx.ext.autosummary',    # Create neat summary tables
    'myst_parser',               # .md files parser - without it it won't accept .md files as a source
    'sphinx_markdown_builder',   # to build markdown files
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_member_order = "groupwise"
autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

templates_path = ['_templates']
exclude_patterns: list[str] = [
    ".build",
    "**.ipynb_checkpoints",
    "**/.pytest_cache",
    "**/.git",
    "**/.github",
    "**/.venv",
    "**/*.egg-info",
    "**/build",
    "**/dist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# html_logo = "_static/images/logos/instant-learn-icon.png"
# html_favicon = "_static/images/logos/instant-learn-favicon.png"
# html_theme_options = {
#     "logo": {
#         "text": "Geti Instant Learn",
#     },
# }

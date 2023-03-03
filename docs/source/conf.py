# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from pkg_resources import get_distribution
import shutil
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'KBMOD'
copyright = '2023, KBMOD Developers'
author = 'KBMOD Developers'

release = get_distribution('kbmod').version
version = '.'.join(release.split('.')[:2])

project_root = os.path.abspath("../../")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_codeautolink',
    'sphinx_gallery.load_style',
    'sphinx_design',
    'numpydoc',
    'nbsphinx'    
]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']

exclude_patterns = ["reference",]

nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static',]

html_logo = "_static/kbmod.svg"

html_theme_options = {
    "navbar_align": "content",
}

html_css_files = ["kbmod.css"]

# Whether to create a Sphinx table of contents for the lists of class methods and attributes.
# If a table of contents is made, Sphinx expects each entry to have a separate page.
numpydoc_class_members_toctree = False


# -- NBSphinx does not support out of root dir linking ----------------------
def all_but_ipynb(dir, contents):
    """Find all files that do not have .ipynb extension."""
    result = []
    for c in contents:
        if "reference" in c:
            # skip the subdir
            result.append(c)
        elif os.path.isfile(os.path.join(dir,c)) and (not c.endswith(".ipynb")):
            result.append(c)
    return result


examples_root = os.path.join(project_root, "docs/source/examples")
shutil.rmtree(os.path.join(examples_root, "_notebooks"), ignore_errors=True)
shutil.copytree(os.path.join(project_root, "notebooks"),
                os.path.join(examples_root, "_notebooks"),
                ignore=all_but_ipynb)


"""Sphinx configuration for sklearn-decision."""
from __future__ import annotations

import sklearn_decision

project = "sklearn-decision"
author = "Jake Timothy"
copyright = "2026, Jake Timothy"
release = sklearn_decision.__version__
version = release

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "**.ipynb_checkpoints", "auto_examples/*.md5", "auto_examples/*.json"]

# Markdown pages (the guide, the design doc, the changelog)
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# API reference
templates_path = ["_templates"]
autosummary_generate = True
autodoc_default_options = {"members": False}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Example gallery. The examples read model answers from the answer caches
# committed under benchmarks/cache, so the docs build needs no API key, no
# GPU and no model download; a cache miss fails the build instead of spending.
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"[\\/]plot_",  # both path separators, so examples run on Windows too
    "abort_on_example_error": True,
    "download_all_examples": False,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}

html_theme = "pydata_sphinx_theme"
html_title = "sklearn-decision"
html_baseurl = "https://jaketimothy.github.io/sklearn-decision/"  # canonical links for search engines
html_theme_options = {
    "github_url": "https://github.com/jaketimothy/sklearn-decision",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}
html_context = {"default_mode": "auto"}

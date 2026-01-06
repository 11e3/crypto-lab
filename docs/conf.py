"""
Sphinx configuration file for Upbit Quant System API documentation.

This file configures Sphinx to automatically generate API documentation
from Python docstrings using autodoc.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 프로젝트 정보
project = "Upbit Quant System"
copyright = "2026, Upbit Quant System"
author = "Upbit Quant System"
release = "0.1.0"
version = "0.1.0"

# Sphinx 확장
extensions = [
    "sphinx.ext.autodoc",  # 자동 문서 생성
    "sphinx.ext.autosummary",  # 자동 요약
    "sphinx.ext.viewcode",  # 소스 코드 링크
    "sphinx.ext.napoleon",  # Google/NumPy 스타일 docstring 지원
    "sphinx.ext.intersphinx",  # 다른 프로젝트 문서 참조
    "sphinx.ext.todo",  # TODO 항목 표시
    "sphinx.ext.coverage",  # 문서 커버리지
    "sphinx_rtd_theme",  # Read the Docs 테마
]

# autodoc 설정
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "special-members": "__init__, __repr__",
}

# autosummary 설정
autosummary_generate = True
autosummary_imported_members = True

# Napoleon 설정 (Google/NumPy 스타일 docstring)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# HTML 출력 설정
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# 정적 파일 경로 (디렉토리가 없으면 빈 리스트)
import os
static_path = "_static"
if os.path.exists(static_path):
    html_static_path = [static_path]
    html_css_files = ["custom.css"]
else:
    html_static_path = []
    html_css_files = []

# HTML 출력 경로
html_output_dir = "_build/html"

# 소스 파일 위치
templates_path = ["_templates"]

# 제외할 파일/디렉토리
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",  # README는 별도로 처리하지 않음
    "README_SPHINX.md",  # Sphinx 가이드는 제외
    "api/*.md",  # .md 파일은 .rst 파일과 중복되므로 제외
]

# 마크다운 파일 처리
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# 마크다운 파서 (myst-parser가 설치되어 있다면)
try:
    import myst_parser

    extensions.append("myst_parser")
    source_suffix[".md"] = "markdown"
except ImportError:
    pass

# intersphinx 매핑 (다른 프로젝트 문서 참조)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# TODO 확장 설정
todo_include_todos = True

# 문서 커버리지 설정
coverage_show_missing_items = True

# 언어 설정
language = "ko"

# 인코딩
source_encoding = "utf-8"

#!/usr/bin/env python3

from setuptools import setup  # type: ignore


long_description = """============================================================
    UP_SKDECIDE
 ============================================================

    up_skdecide is a small package that allows an exchange of
    equivalent data structures between unified_planning and scikit-decide.
"""

setup(
    name="up_skdecide",
    version="0.0.1",
    description="up_skdecide",
    author="AIPlan4EU Organization",
    author_email="aiplan4eu@fbk.eu",
    url="https://www.aiplan4eu-project.eu",
    packages=["up_skdecide"],
    install_requires=[
        "unified-planning>=1.1.0",
        "scikit-decide>=1.0.0",
        "simplejson==3.17.6",
    ],
    license="APACHE",
)

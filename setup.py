# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

VERSION = {}  # type: ignore
with open("cybo/version.py", "r", encoding="utf-8") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name='cybo',
    version=VERSION["VERSION"],
    author='kebo',
    author_email='kebo0912@outlook.com',
    url='https://github.com/bo-ke/cybo',
    description='an nlp training framework base tf2.0',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.3.0'
    ],
    # entry_points={"console_scripts": ["cybo=cybo.__main__:run"]},
    python_requires='>=3.6.1',
)

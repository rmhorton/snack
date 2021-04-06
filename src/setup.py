
# python setup.py bdist_wheel

import sys
import codecs

from setuptools import setup, find_packages

import snack

install_requires = [
    'networkx>=2.5',
    'python-louvain>=0.15',
]

setup(
    name='snack',
    version=snack.__version__,
    description=snack.__doc__.strip(),
    # long_description=long_description(),
    # long_description_content_type='text/x-rst',
)
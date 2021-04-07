
# python setup.py bdist_wheel
# pip install --force-reinstall --user dist\snack-0.0.1-py3-none-any.whl

import io
from setuptools import setup, find_packages

# import snack

with io.open('README.md', encoding='utf_8') as fp:
    readme = fp.read()

setup(
    name='snack',
    version='0.0.1', # snack.__version__,
    description='feature engineering functions for EHR data', # snack.__doc__.strip(),
    author= 'Robert Horton',
    url='https://github.com/rmhorton/snack',
    long_description=readme,
    long_description_content_type='text/markdown; charset=UTF-8',
    packages=find_packages(),
    install_requires = ['networkx>=2.5', 'python-louvain>=0.15']
)

# pip install pydoc-markdown
# conda install mkdocs
# pydoc-markdown --bootstrap mkdocs
# -- generate pydoc-markdown.yml
# pydoc-markdown -m snack.snack --render-toc > snack_doc.md
# -- generate function-level documentation

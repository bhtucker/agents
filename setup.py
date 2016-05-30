"""
ABM
~~~~~~~

Agent based modeling
"""

from setuptools import setup, find_packages


def get_requirements(suffix=''):
    with open('requirements%s.txt' % suffix) as f:
        result = f.read().splitlines()
    return result

setup(
    name='ABM',
    version='0.0.1',
    url='https://github.com/bhtucker/agents',
    author='Benson Tucker',
    author_email='bensontucker@gmail.com',
    description='Agent based modeling',
    long_description='',
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    platforms='any')

from setuptools import setup, find_packages
from beagle import __version__ as VERSION

readme = 'README.md'
long_description = open( readme ).read()

config = {
    'description': 'beagle - A lightweight genetic algorithm framework',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'author': 'Benjamin J. Morgan',
    'author_email': 'b.j.morgan@bath.ac.uk',
    'url': 'https://github.com/bjmorgan/beagle',
    'download_url': "https://github.com/bjmorgan/bsym/archive/%s.tar.gz" % (VERSION),
    'author_email': 'b.j.morgan@bath.ac.uk',
    'version': VERSION,
    'install_requires': [ 'numpy',
                          'scipy',
                          'coverage',
                          'codeclimate-test-reporter' ],
    'python_requires': '>=3.5',
    'license': 'MIT',
    'packages': [ 'beagle', 'beagle.interface' ],
    'scripts': [],
    'name': 'beagle'
}

setup(**config)

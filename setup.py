"""
vificov setup.

For development installation:
    pip install -e /path/to/vificov
"""

from setuptools import setup

with open('README.rst') as f:
    long_description = f.read()

setup(name='vificov',
      version='1.0.3',
      description=('Visual Field Coverage (ViFiCov) visualization in python.'),
      url='https://github.com/MSchnei/vificov',
      author='Marian Schneider',
      author_email='marian.schneider@maastrichtuniversity.nl',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'matplotlib', 'nibabel'],
      keywords=['pRF', 'fMRI', 'retinotopy'],
      long_description=long_description,
      packages=['vificov'],
      entry_points={
          'console_scripts': [
              'vificov = vificov.__main__:main',
              ]},
      )

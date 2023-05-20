from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
import versioneer

DISTNAME = 'lpocv'
VERSION = versioneer.get_version()
CMDCLASS = versioneer.get_cmdclass()
DESCRIPTION = 'Leave-pair-out cross-validation'
LONG_DESCRIPTION = '''
**lpocv** is a Python module that implements leave-pair-out cross-validation 
for machine learning classification and regression problems. It allows the user 
to match the samples by a confounding batch variable and also include errors 
on the labels (for regression problems).
'''
KEYWORDS = ['leave-pair-out', 'cross-validation', 'batch effects', 'bioinformatics']
AUTHOR = 'Maulik Nariya'
AUTHOR_EMAIL = 'mauliknariya@gmail.com'
URL = 'http://github.com/mauliknariya/lpocv'
PACKAGES = find_packages()
INSTALL_REQUIRES = ['numpy', 'scikit_learn']
CLASSIFIERS = ['Development Status :: 3 - Alpha', 
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: MIT License',
                'Natural Language :: English',
                'Operating System :: OS Independent',
                'Programming Language :: Python :: 3',
                'Topic :: Scientific/Engineering :: Bio-Informatics'
                ]

setup(name=DISTNAME,
      version=VERSION,
      cmdclass=CMDCLASS,
      description=DESCRIPTION,
      keywords=KEYWORDS,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      packages=PACKAGES,
      install_requires=INSTALL_REQUIRES,
      classifiers=CLASSIFIERS 
      )

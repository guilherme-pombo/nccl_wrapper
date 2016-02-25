#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc. All rights reserved.
# Unauthorized copying or viewing of this file outside Nervana Systems Inc.,
# via any medium is strictly prohibited. Proprietary and confidential.
# ----------------------------------------------------------------------------

import os
from setuptools import setup, find_packages
import subprocess

# Define version information
VERSION = '0.0.1'
FULLVERSION = VERSION
write_version = True

try:
    pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                            stdout=subprocess.PIPE)
    (so, serr) = pipe.communicate()
    if pipe.returncode == 0:
        FULLVERSION += "+%s" % so.strip().decode("utf-8")
except:
    pass

setup(name='ncclcomm',
      version=VERSION,
      description="Wrapper for nccl library",
      long_description=open('README.md').read(),
      author='Nervana Systems',
      author_email='info@nervanasys.com',
      url='http://www.nervanasys.com',
      packages=find_packages(exclude=["tests"]),
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Environment :: Console :: Curses',
                   'Intended Audience :: Developers',
                   'License :: Proprietary License',
                   'Operating System :: POSIX',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: ' +
                   'Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: System :: Distributed Computing'])

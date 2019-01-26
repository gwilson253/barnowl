# -*- coding: utf-8 -*-

from setuptools import setup
import sys

def forbid_publish():
    argv = sys.argv
    blacklist = ['register', 'upload']

    for command in blacklist:
        if command in argv:
            values = {'command': command}
            print('Command "%(command)s" has been blacklisted, exiting...' %
                  values)
            sys.exit(2)

forbid_publish()

setup(name='ml_deploy',
      version='0.1',
      description='pseudo-production platform for running & evaluating ML models',
      url='https://github.com/inside-track/analytics/tree/master/ml-deploy',
      author='InsideTrack | Greg Wilson',
      author_email='greg.wilson@insidetrack.com',
      license='MIT',
      packages=['ml_deploy'],
      install_requires=['sklearn',
                        'numpy',
                        'pandas',
                        'sqlalchemy',
                        'boto'
                        ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest']
      )
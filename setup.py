#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

try:
    from setuptools import setup
except ImportError:
    from distutils.setuptools import setup

import gestrecog
import os

setup(
    name=gestrecog.__title__,
    version=gestrecog.__version__,
    description='Gesture recognition based on Kinect. Y. T. Chung\'s graduation design',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.rst')).read(),
    license=gestrecog.__license__,
    platforms=[
        'Operating System :: MacOS :: Mac OS X',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: BSD',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
    ],
    author=gestrecog.__author__,
    author_email='zonyitoo@gmail.com',
    url='https://github.com/zonyitoo/GestureRecognition',
    packages=[
        'gestrecog',
    ],
    keywords='Kinect OpenCV OpenNI Gesture Recognition',
    include_package_data=True,
    install_requires=[
        'cv2>=3.0.0',
        'numpy>=1.8.0',
        'scipy>=0.11.0',
    ],
    entry_points={
        'console_scripts': [
            'gestrecog=gestrecog.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        gestrecog.__license__,
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS :: Mac OS X',
        'Operating System :: Microsoft :: Windows 7'
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)

-------------------------------
Gesture Recognition with Kinect
-------------------------------

*Just For GRADUATION.*

Requirements
============

* Unix/Linux, Microsoft Windows®

* OpenNI 1.x, OpenNI 2.x (Currently OpenNI 2.x only support Windows® Kinect SDK)

* OpenCV >= 2.4.7

* Microsoft® Kinect

Building
========

* OS X 下打开 ``GestureRecognition.xcodeproj``，设置好 OpenCV 的库路径，直接运行即可

* Linux/Unix

.. code:: bash

    $ g++ *.h *.cpp -o gesture-recognition `pkg-config opencv --libs --cflags`
    $ ./gesture-recognition

ScreenShots
===========

*Will be uploaded after first BETA*

Thanks
======

* Directed by `Assiciate Prof. Qingge Ji <http://sist.sysu.edu.cn/main/default/teainfo.aspx?id=73&no=1&pId=10>`_

-------------------------------
Gesture Recognition with Kinect
-------------------------------

Just for **GRADUATION**.

Aims
====

* Skin detection. ✔︎
* Obtain one hand's contour. ✔︎
* Distinguish hand from head. ✔︎
* Mark finger tips and palm's positions. ✔︎
* Use K-Means algorithm to detect finger tips' positions. ✖︎
* Use bounding polygon to detect finger tips' positions. ✔︎
* Remove points on arm. ✔︎
* Follow finger tips and palm's positions change.
* Use finite-state machine to define and recognize hand gesture.
* Improve precision of finger tips and palm's positions.
* Be able to follow finger tips's movement.
* Improve performance, avaliability and usability.

Requirements
============

* Unix/Linux, Microsoft Windows®
* OpenNI 1.x, OpenNI 2.x (Currently OpenNI 2.x only support Windows® Kinect SDK)
* OpenCV >= 2.4.7
* Microsoft® Kinect
* C++ compiler with C++11 standard support

Building
========

* OS X: Open ``GestureRecognition.xcodeproj`` with Xcode, then press ``⌘`` + ``R``.
* Linux/Unix

.. code:: bash

    $ cd GestureRecognition
    $ g++ -std=c++11 *.h *.cpp -o gesture-recognition `pkg-config opencv --libs --cflags`
    $ ./gesture-recognition

ScreenShots
===========

*Will be uploaded after first BETA released*

Thanks
======

* Directed by `Associate Prof. Qingge Ji <http://sist.sysu.edu.cn/main/default/teainfo.aspx?id=73&no=1&pId=10>`_
* `Open Source Computer Vision Library <https://github.com/Itseez/opencv>`_

License
=======

`BSD 3-Clause License <http://opensource.org/licenses/BSD-3-Clause>`_

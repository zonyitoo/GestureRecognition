-----------------------
基于 Kinect 的手势识别程序
-----------------------

苦逼的毕设 😂

Requirements
============

* Unix/Linux, Windows

* OpenNI, OpenNI2 (由于 OpenNI2 在 Unix/Linux 环境下对 Kinect 支持不完善，建议在 Windows 下配合 Microsoft Kinect SDK 使用）

* OpenCV

* Kinect

Building
========

* OS X 下打开 `GestureRecognition.xcodeproj`，设置好 OpenCV 的库路径，直接运行即可

* Linux 

.. code: bash

    $ g++ *.h *.cpp -o gesture-recognition `pkg-config opencv --libs --cflags`
    
ScreenShots
===========

*在初版完成后上传*
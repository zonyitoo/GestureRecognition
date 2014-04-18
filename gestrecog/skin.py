# -*- coding:utf-8 -*-

from __future__ import unicode_literals, print_function

import cv2


class SkinDetector(object):

    def __init__(self,
                 median_blur_kernel_size=11,
                 gaussian_blur_kernel_size=11,
                 structuring_elem_size=(5, 5)):
        self.median_blur_kernel_size = median_blur_kernel_size
        self.gaussian_blur_kernel = (gaussian_blur_kernel_size, gaussian_blur_kernel_size)
        self.structuring_elem_size = structuring_elem_size

    def generate_mask_from_bgr(self, bgr):
        bluredbgr = cv2.GaussianBlur(bgr, self.gaussian_blur_kernel, 0)
        cv2.medianBlur(bluredbgr, self.median_blur_kernel_size, dst=bluredbgr)
        cv2.cvtColor(bluredbgr, cv2.COLOR_BGR2HSV, dst=bluredbgr)
        hsvmask1 = cv2.inRange(bluredbgr, (0, 30, 30, 0), (40, 170, 255, 0))
        hsvmask2 = cv2.inRange(bluredbgr, (156, 30, 30, 0), (180, 170, 255, 0))
        hsvmask = cv2.bitwise_or(hsvmask1, hsvmask2)

        struc = cv2.getStructuringElement(cv2.MORPH_RECT, self.structuring_elem_size)
        cv2.erode(hsvmask, struc)
        cv2.morphologyEx(hsvmask, cv2.MORPH_OPEN, struc)
        cv2.dilate(hsvmask, struc)
        cv2.morphologyEx(hsvmask, cv2.MORPH_CLOSE, struc)

        cv2.GaussianBlur(hsvmask, (3, 3), 0, hsvmask)

        return hsvmask

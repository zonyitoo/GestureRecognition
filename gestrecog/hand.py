# -*- coding:utf-8 -*-
from __future__ import unicode_literals, print_function
import cv2
from scipy.signal import argrelextrema, argrelmax
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np


def show_histogram(hist, binsize, step=1, winname="HIST"):
    # print peaks
    STEP = step
    pts = np.column_stack((np.arange(binsize * STEP, step=STEP).reshape(binsize, 1), hist))
    histImg = np.zeros((300, binsize * STEP), dtype='uint8')
    cv2.polylines(histImg, [pts], False, (255, 0, 0))
    histImg = np.flipud(histImg)
    cv2.imshow(winname, histImg)


class HandFeatureFindingError(RuntimeError):

    def __init__(self, msg):
        super(HandFeatureFindingError, self).__init__(msg)


class HandFeatureFinder(object):

    def __init__(self):
        pass

    def get_color_histogram(self, depth, gkernel=(3, 3)):
        BIN_SIZE = 256
        RANGES = (0, 255)

        hist = cv2.calcHist(images=(depth,), channels=(0,),
                            mask=None, histSize=(BIN_SIZE,), ranges=RANGES)
        hist[0][0] = 0
        cv2.normalize(hist, hist, 0, BIN_SIZE - 1, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist))
        hist = gaussian_filter(hist, gkernel)
        for i in xrange(1, len(hist)):
            if hist[i][0] == 0 and hist[i - 1][0] <= 0:
                hist[i][0] = hist[i - 1][0] - 1
        show_histogram(hist, BIN_SIZE, step=3)
        #hist = median_filter(hist, (3, 3))
        peaks, indeces = argrelextrema(hist, np.greater_equal)
        #peaks = argrelmax(hist, order=winmidsize)[0]

        return hist, peaks

    def filter_depth_by_histogram_peaks(self, depth, peaks, threshold=10):
        if len(peaks) < 1:
            raise HandFeatureFindingError('Error while processing histogram')
        depthmask = cv2.inRange(depth, lowerb=(int(peaks[-1] - threshold), ), upperb=(int(peaks[-1] + threshold), ))
        struc = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cv2.morphologyEx(depthmask, cv2.MORPH_OPEN, struc)
        depthfilt = cv2.add(depth, 0, mask=depthmask)
        cv2.imshow('DEPTH FILTED', depthfilt)
        return depthfilt

    def analyze_palm(self, depthfilt):
        distMap = np.zeros(depthfilt.shape, dtype=np.dtype('uint8'))
        # for cont in contours:
        #     if cv2.contourArea(cont) < 1000:
        #         continue
        depth8u = np.vectorize(lambda x: 255 if x > 0 else 0, otypes=['uint8'])(depthfilt)
        # cv2.drawContours(depth8u, [cont, ], -1, [255], cv2.FILLED)
        outp = cv2.distanceTransform(depth8u, cv2.DIST_L2, cv2.DIST_MASK_5)

        i = outp.argmax()
        max_index = np.unravel_index(i, outp.shape)
        max_index = max_index[1], max_index[0]
        max_val = outp[max_index[1]][max_index[0]]

        finger_struc = cv2.getStructuringElement(cv2.MORPH_RECT, (int(max_val / 4.5), int(max_val / 4.5)))
        cv2.morphologyEx(outp, cv2.MORPH_CLOSE, finger_struc, dst=outp)

        convt = np.uint8(outp)
        distMap = cv2.add(6 * convt, distMap)

        # max_index as palm_pos
        palm_pos = np.array(max_index)

        # Find palm
        palm_struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(max_val / 3), int(max_val / 3)))
        distpalm = cv2.erode(outp, palm_struc)
        cv2.morphologyEx(distpalm, cv2.MORPH_OPEN, palm_struc, dst=distpalm)
        cv2.morphologyEx(distpalm, cv2.MORPH_CLOSE, palm_struc, dst=distpalm)
        # cv2.dilate(distpalm, palm_struc, dst=distpalm)

        # Finger mask
        transpmask = np.vectorize(lambda x: 0 if x > 0 else 255, otypes=[np.uint8])
        distfmask = transpmask(distpalm)
        # cv2.imshow("DIST_F_MASK", distfmask)

        finger = cv2.add(depth8u, 0, mask=distfmask)
        cv2.imshow("FINGER", finger)
        edge_struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(max_val / 3), int(max_val / 3)))
        cv2.morphologyEx(finger, cv2.MORPH_OPEN, edge_struc, dst=finger)
        # cv2.imshow("FINGER", finger)

        # Find finger contours
        _, finger_contours, _ = cv2.findContours(finger, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(handimg, finger_contours, -1, (0, 255))

        fingers = []
        for cont in finger_contours:
            if len(cont) < 3:
                continue
            hull = cv2.convexHull(cont, returnPoints=True)

            np.append(hull, hull[0])

            max_dist = 0.0
            max_pnt1, max_pnt2 = None, None
            q = 1
            for i, pnt in enumerate(hull):
                if i == len(hull) - 1:
                    break
                while np.cross(hull[i + 1][0] - pnt[0], hull[(q + 1) % len(hull)][0] - pnt[0])\
                        > np.cross(hull[i + 1][0] - pnt[0], hull[q][0] - pnt[0]):
                    q = (q + 1) % len(hull)
                dist1 = np.linalg.norm(pnt[0] - hull[q][0])
                dist2 = np.linalg.norm(hull[i + 1][0] - hull[(q + 1) % len(hull)][0])
                mdist = 0.0
                mpnt = None
                if dist1 < dist2:
                    mdist = dist2
                    mpnt = hull[i + 1][0], hull[(q + 1) % len(hull)][0]
                else:
                    mdist = dist1
                    mpnt = pnt[0], hull[q][0]
                if max_dist < mdist:
                    max_dist = mdist
                    max_pnt1, max_pnt2 = mpnt

            fingers.append((max_pnt1, max_pnt2))

        tmpfing = []
        for ind, fin in enumerate(fingers):
            p1, p2 = fin
            d1 = np.linalg.norm(p1 - palm_pos)
            d2 = np.linalg.norm(p2 - palm_pos)
            if d1 > max_val * 2.5 and d2 > max_val * 2.5:
                continue
            if d1 > d2:
                tmpfing.append((p2, p1))
            else:
                tmpfing.append((p1, p2))

        fingers = tmpfing

        def compare(x, y):
            cr = np.cross(x[0] - palm_pos, y[0] - palm_pos)
            if cr != 0:
                return int(cr - 0)
            return int(np.linalg.norm(x[0] - palm_pos) - np.linalg.norm(y[0] - palm_pos))
        fingers.sort(cmp=compare, reverse=True)

        return palm_pos, fingers

    def get_features(self, depth, depth_orig):
        _, peaks = self.get_color_histogram(depth)
        depthfilt = self.filter_depth_by_histogram_peaks(depth, peaks)
        palm_2d, fingers_2d = self.analyze_palm(depthfilt)

        palm_3d = tuple(palm_2d) + (depth[palm_2d[0]][palm_2d[1]], )
        fingers_3d = [(tuple(finger[0]) + (depth_orig[finger[0][0]][finger[0][1]], ),
                       tuple(finger[1]) + (depth_orig[finger[1][0]][finger[1][1]], ))
                      for finger in fingers_2d]
        return palm_3d, fingers_3d

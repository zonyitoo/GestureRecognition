#!/usr/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import unicode_literals, print_function

import cv2
import numpy as np
from scipy.signal import argrelextrema, argrelmax
from scipy.ndimage import gaussian_filter, median_filter

capture = cv2.VideoCapture()
capture.open(cv2.CAP_OPENNI)


def get_hsvmask(bgr):
    bluredbgr = cv2.GaussianBlur(bgr, (11, 11), 0)
    cv2.medianBlur(bluredbgr, 11, dst=bluredbgr)
    cv2.cvtColor(bluredbgr, cv2.COLOR_BGR2HSV, dst=bluredbgr)
    hsvmask1 = cv2.inRange(bluredbgr, (0, 30, 30, 0), (40, 170, 255, 0))
    hsvmask2 = cv2.inRange(bluredbgr, (156, 30, 30, 0), (180, 170, 255, 0))
    hsvmask = cv2.bitwise_or(hsvmask1, hsvmask2)

    struc = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cv2.erode(hsvmask, struc)
    cv2.morphologyEx(hsvmask, cv2.MORPH_OPEN, struc)
    cv2.dilate(hsvmask, struc)
    cv2.morphologyEx(hsvmask, cv2.MORPH_CLOSE, struc)

    cv2.GaussianBlur(hsvmask, (3, 3), 0, hsvmask)

    return hsvmask


def show_histogram(hist, binsize, step=1, winname="HIST"):
    # print peaks
    STEP = step
    pts = np.column_stack((np.arange(binsize * STEP, step=STEP).reshape(binsize, 1), hist))
    histImg = np.zeros((300, binsize * STEP), dtype='uint8')
    cv2.polylines(histImg, [pts], False, (255, 0, 0))
    histImg = np.flipud(histImg)
    cv2.imshow(winname, histImg)


def get_histogram(depth):
    BIN_SIZE = 256
    RANGES = (0, 255)
    cv2.imshow("DEPTH", depth)

    hist = cv2.calcHist(images=(depth,), channels=(0,),
                        mask=None, histSize=(BIN_SIZE,), ranges=RANGES)
    hist[0][0] = 0
    cv2.normalize(hist, hist, 0, BIN_SIZE - 1, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist))
    hist = gaussian_filter(hist, (3, 3))
    for i in xrange(1, len(hist)):
        if hist[i][0] == 0 and hist[i - 1][0] <= 0:
            hist[i][0] = hist[i - 1][0] - 1
    #hist = median_filter(hist, (3, 3))
    peaks, indeces = argrelextrema(hist, np.greater_equal)
    #peaks = argrelmax(hist, order=winmidsize)[0]

    show_histogram(hist, BIN_SIZE)
    return hist, peaks


def get_depthfilt(depth, peaks, threshold=10):
    if len(peaks) < 1:
        return None
    depthmask = cv2.inRange(depth, lowerb=(int(peaks[-1] - threshold), ), upperb=(int(peaks[-1] + threshold), ))
    struc = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cv2.morphologyEx(depthmask, cv2.MORPH_OPEN, struc)
    depthfilt = cv2.add(depth, 0, mask=depthmask)
    return depthfilt


def analyze_palm(depthfilt):
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

if __name__ == '__main__':
    import timeit

    while capture.grab():
        start_time = timeit.default_timer()
        _, bgr = capture.retrieve(flag=cv2.CAP_OPENNI_BGR_IMAGE)
        _, depth = capture.retrieve(flag=cv2.CAP_OPENNI_DISPARITY_MAP)
        _, validMask = capture.retrieve(flag=cv2.CAP_OPENNI_VALID_DEPTH_MASK)

        hsvmask = get_hsvmask(bgr)
        depth = cv2.add(depth, 0, mask=hsvmask)
        # cv2.GaussianBlur(depth, (3, 3), 0, dst=depth)
        cv2.medianBlur(depth, 5, dst=depth)
        # cv2.imshow("MASKED DEPTH", depth)

        _, peaks = get_histogram(depth)
        depthfilt = get_depthfilt(depth, peaks)
        cv2.imshow("DEPTH FILT", depthfilt)
        if depthfilt is None:
            if cv2.waitKey(300) == 27:
                break
            continue

        palm_pos, fingers = analyze_palm(depthfilt)
        handimg = np.zeros(bgr.shape, dtype=np.dtype('float32'))
        cv2.circle(handimg, tuple(palm_pos), 20, (255, 255, 0, 0), 5)

        finger_colors = (
            (0, 255, 0),  # G
            (0, 0, 255),  # R
            (255, 0, 0),  # B
            (255, 255, 0),  # CYAN
            (0, 255, 255),  # YELLOW
        )
        for ind, fin in enumerate(fingers):
            p1, p2 = fin
            cv2.line(handimg, tuple(palm_pos), tuple(p1), finger_colors[ind % len(finger_colors)], thickness=3)
            cv2.line(handimg, tuple(p1), tuple(p2), finger_colors[ind % len(finger_colors)], thickness=3)
            cv2.putText(handimg, str(ind), tuple(p2), cv2.FONT_HERSHEY_PLAIN, 3, (255, ), thickness=3)
            # cv2.line(bgr, tuple(palm_pos), tuple(p2), (0, 255, 255), thickness=3)

        # cv2.imshow("DEPTH8U", depth8u)
        cv2.imshow("BGR", bgr)
        cv2.imshow("HAND", handimg)

        stop_time = timeit.default_timer()

        print('Runtime: %sms' % (stop_time - start_time, ))
        if cv2.waitKey(300) == 27:
            break

    capture.release()

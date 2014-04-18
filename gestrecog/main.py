#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals, print_function

import cv2

from hand import HandFeatureFinder
from skin import SkinDetector


def main():
    import timeit
    capture = cv2.VideoCapture()
    capture.open(cv2.CAP_OPENNI)

    skin_detector = SkinDetector()
    feature_finder = HandFeatureFinder()

    while capture.grab():
        start_time = timeit.default_timer()
        _, bgr = capture.retrieve(flag=cv2.CAP_OPENNI_BGR_IMAGE)
        _, depth_orig = capture.retrieve(flag=cv2.CAP_OPENNI_DISPARITY_MAP)
        # _, validMask = capture.retrieve(flag=cv2.CAP_OPENNI_VALID_DEPTH_MASK)

        try:
            skin_mask = skin_detector.generate_mask_from_bgr(bgr)
            filt_depth = cv2.add(depth_orig, 0, mask=skin_mask)
            palm_pos, fingers = feature_finder.get_features(filt_depth)

            print(palm_pos)
            print(fingers)

            cv2.circle(bgr, (palm_pos[0], palm_pos[1]), 20, (255, 255, 0, 0), 5)

            finger_colors = (
                (0, 255, 0),  # G
                (0, 0, 255),  # R
                (255, 0, 0),  # B
                (255, 255, 0),  # CYAN
                (0, 255, 255),  # YELLOW
            )

            for ind, finger in enumerate(fingers):
                p1, p2 = finger

                cv2.line(bgr, (palm_pos[0], palm_pos[1]), (p1[0], p1[1]),
                         finger_colors[ind % len(finger_colors)], thickness=3)
                cv2.line(bgr, (p1[0], p1[1]), (p2[0], p2[1]),
                         finger_colors[ind % len(finger_colors)], thickness=3)
                cv2.putText(bgr, str(ind), (p2[0], p2[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, ), thickness=3)

            stop_time = timeit.default_timer()
            print('Runtime: %fms' % (stop_time - start_time))

            cv2.imshow("BGR", bgr)
        except Exception as e:
            print(e)

        if cv2.waitKey(300) == 27:
            break

    capture.release()


if __name__ == '__main__':
    main()

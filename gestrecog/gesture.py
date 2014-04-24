# -*- coding:utf-8 -*-
from __future__ import print_function, unicode_literals

import cv2
import pickle


class GestureDetector(object):

    def __init__(self):
        self.palm_positions = []
        self.finger_positions = []

    def add_position(self, palm_pos, fingers):
        self.palm_positions.append(palm_pos)
        self.finger_positions.append(fingers)

        self.detect()

    def detect(self):
        raise NotImplementedError()

    def clear(self):
        self.palm_positions = []
        self.finger_positions = []


class StaticGestureDetector(GestureDetector):

    def __init__(self):
        super(StaticGestureDetector, self).__init__()

    def detect(self):
        print('StaticGestureDetector %d fingers' % len(self.finger_positions))
        self.clear()


class DynamicRoutineGestureDetector(GestureDetector):

    MOV_FRONT = 'MOV_FRONT'  # 0
    MOV_BACK = 'MOV_BACK'  # 1
    MOV_LEFT = 'MOV_LEFT'  # 2
    MOV_RIGHT = 'MOV_RIGHT'  # 3
    MOV_UP = 'MOV_UP'  # 4
    MOV_DOWN = 'MOV_DOWN'  # 5
    MOV_FRONT_LEFT = 'MOV_FRONT_LEFT'  # 6
    MOV_FRONT_RIGHT = 'MOV_FRONT_RIGHT'  # 7
    MOV_FRONT_UP = 'MOV_FRONT_UP'  # 8
    MOV_FRONT_DOWN = 'MOV_FRONT_DOWN'  # 9
    MOV_FRONT_LEFT_UP = 'MOV_FRONT_LEFT_UP'
    MOV_FRONT_LEFT_DOWN = 'MOV_FRONT_LEFT_DOWN'
    MOV_FRONT_RIGHT_UP = 'MOV_FRONT_RIGHT_UP'
    MOV_FRONT_RIGHT_DOWN = 'MOV_FRONT_RIGHT_DOWN'
    MOV_BACK_LEFT = 'MOV_BACK_LEFT'  # 10
    MOV_BACK_RIGHT = 'MOV_BACK_RIGHT'  # 11
    MOV_BACK_UP = 'MOV_BACK_UP'  # 12
    MOV_BACK_DOWN = 'MOV_BACK_DOWN'  # 13
    MOV_BACK_LEFT_UP = 'MOV_BACK_LEFT_UP'
    MOV_BACK_LEFT_DOWN = 'MOV_BACK_LEFT_DOWN'
    MOV_BACK_RIGHT_UP = 'MOV_BACK_RIGHT_UP'
    MOV_BACK_RIGHT_DOWN = 'MOV_BACK_RIGHT_DOWN'
    MOV_LEFT_UP = 'MOV_LEFT_UP'  # 14
    MOV_LEFT_DOWN = 'MOV_LEFT_DOWN'  # 15
    MOV_RIGHT_UP = 'MOV_RIGHT_UP'  # 16
    MOV_RIGHT_DOWN = 'MOV_RIGHT_DOWN'  # 17
    MOV_STAY = 'MOV_STAY'  # 18

    GEST_NONE = 'GEST_NONE'
    GEST_1 = 'GEST1'  # 上
    GEST_2 = 'GEST2'  # 下
    GEST_3 = 'GEST3'  # 左
    GEST_4 = 'GEST4'  # 右
    GEST_5 = 'GEST5'  # 前
    GEST_6 = 'GEST6'  # 后
    GEST_7 = 'GEST7'  # 矩形
    GEST_8 = 'GEST8'  # 三角形
    GEST_9 = 'GEST9'  # M形

    # dynamic_gestures = {
    #     MOV_FRONT: {
    #         'gest': GEST_5,
    #     },
    #     MOV_BACK: {
    #         'gest': GEST_6,
    #     },
    #     MOV_LEFT: {
    #         'gest': GEST_3,
    #         MOV_DOWN: {
    #             'gest': GEST_NONE,
    #             MOV_RIGHT: {
    #                 'gest': GEST_NONE,
    #                 MOV_UP: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         },
    #         MOV_UP: {
    #             'gest': GEST_NONE,
    #             MOV_RIGHT: {
    #                 'gest': GEST_NONE,
    #                 MOV_DOWN: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         },
    #         MOV_RIGHT_UP: {
    #             'gest': GEST_NONE,
    #             MOV_RIGHT_DOWN: {
    #                 'gest': GEST_8,
    #             },
    #             MOV_UP: {
    #                 'gest': GEST_NONE,
    #                 MOV_RIGHT: {
    #                     'gest': GEST_NONE,
    #                     MOV_DOWN: {
    #                         'gest': GEST_7,
    #                     }
    #                 }
    #             },
    #         },
    #         MOV_RIGHT_DOWN: {
    #             'gest': GEST_NONE,
    #             MOV_RIGHT_UP: {
    #                 'gest': GEST_8,
    #             },
    #             MOV_DOWN: {
    #                 'gest': GEST_NONE,
    #                 MOV_RIGHT: {
    #                     'gest': GEST_NONE,
    #                     MOV_UP: {
    #                         'gest': GEST_7,
    #                     }
    #                 }
    #             },
    #         },

    #     },
    #     MOV_RIGHT: {
    #         'gest': GEST_4,
    #         MOV_DOWN: {
    #             'gest': GEST_NONE,
    #             MOV_LEFT: {
    #                 'gest': GEST_NONE,
    #                 MOV_UP: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         },
    #         MOV_UP: {
    #             'gest': GEST_NONE,
    #             MOV_LEFT: {
    #                 'gest': GEST_NONE,
    #                 MOV_DOWN: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         },
    #         MOV_LEFT_UP: {
    #             'gest': GEST_NONE,
    #             MOV_LEFT_DOWN: {
    #                 'gest': GEST_8,
    #             },
    #             MOV_UP: {
    #                 'gest': GEST_NONE,
    #                 MOV_LEFT: {
    #                     'gest': GEST_NONE,
    #                     MOV_DOWN: {
    #                         'gest': GEST_7,
    #                     }
    #                 }
    #             },
    #         },
    #         MOV_LEFT_DOWN: {
    #             'gest': GEST_NONE,
    #             MOV_LEFT_UP: {
    #                 'gest': GEST_8,
    #             },
    #             MOV_DOWN: {
    #                 'gest': GEST_NONE,
    #                 MOV_LEFT: {
    #                     'gest': GEST_NONE,
    #                     MOV_UP: {
    #                         'gest': GEST_7,
    #                     }
    #                 }
    #             },
    #         },
    #     },
    #     MOV_UP: {
    #         'gest': GEST_1,
    #         MOV_LEFT: {
    #             'gest': GEST_NONE,
    #             MOV_DOWN: {
    #                 'gest': GEST_NONE,
    #                 MOV_RIGHT: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         },
    #         MOV_RIGHT: {
    #             'gest': GEST_NONE,
    #             MOV_DOWN: {
    #                 'gest': GEST_NONE,
    #                 MOV_LEFT: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         }
    #     },
    #     MOV_DOWN: {
    #         'gest': GEST_2,
    #         MOV_LEFT: {
    #             'gest': GEST_NONE,
    #             MOV_UP: {
    #                 'gest': GEST_NONE,
    #                 MOV_RIGHT: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         },
    #         MOV_RIGHT: {
    #             'gest': GEST_NONE,
    #             MOV_UP: {
    #                 'gest': GEST_NONE,
    #                 MOV_LEFT: {
    #                     'gest': GEST_7,
    #                 }
    #             }
    #         }
    #     },
    #     MOV_FRONT_LEFT: {
    #         'gest': GEST_3,
    #     },
    #     MOV_FRONT_RIGHT: {
    #         'gest': GEST_4,
    #     },
    #     MOV_FRONT_UP: {
    #         'gest': GEST_NONE,
    #     },
    #     MOV_FRONT_DOWN: {
    #         'gest': GEST_NONE,
    #     },
    #     MOV_BACK_LEFT: {
    #         'gest': GEST_NONE,
    #     },
    #     MOV_BACK_RIGHT: {
    #         'gest': GEST_NONE,
    #     },
    #     MOV_BACK_UP: {
    #         'gest': GEST_NONE,
    #     },
    #     MOV_BACK_DOWN: {
    #         'gest': GEST_NONE,
    #     },
    #     MOV_LEFT_UP: {
    #         'gest': GEST_NONE,
    #         MOV_LEFT_DOWN: {
    #             'gest': GEST_NONE,
    #             MOV_RIGHT: {
    #                 'gest': GEST_8,
    #             }
    #         }
    #     },
    #     MOV_LEFT_DOWN: {
    #         'gest': GEST_NONE,
    #         MOV_LEFT_UP: {
    #             'gest': GEST_NONE,
    #             MOV_RIGHT: {
    #                 'gest': GEST_8,
    #             }
    #         }
    #     },
    #     MOV_RIGHT_UP: {
    #         'gest': GEST_NONE,
    #         MOV_RIGHT_DOWN: {
    #             'gest': GEST_NONE,
    #             MOV_LEFT: {
    #                 'gest': GEST_8,
    #             }
    #         }
    #     },
    #     MOV_RIGHT_DOWN: {
    #         'gest': GEST_NONE,
    #         MOV_RIGHT_UP: {
    #             'gest': GEST_NONE,
    #             MOV_LEFT: {
    #                 'gest': GEST_8,
    #             }
    #         }
    #     },
    #     'gest': GEST_NONE,
    # }

    def get_move(self, prev, curr):
        import math
        px, py, pz = prev
        cx, cy, cz = curr
        deltx = px - cx
        delty = py - cy
        deltz = cz - pz
        deltz = deltz if deltz < 128 else deltz - 255

        dist3d = deltx ** 2 + delty ** 2 + deltz ** 2
        if dist3d < 15 ** 2:
            return self.MOV_STAY

        dist2d = deltx ** 2 + delty ** 2

        Z_THRESHOLD = 8

        if dist2d > 15 ** 2:
            sin2d = delty / math.sqrt(dist2d)
            ang2d = math.asin(sin2d) / math.pi * 180.0

            angrange = [i * 22.5 - 90 for i in xrange(9)]
            if ang2d >= angrange[0] and ang2d < angrange[1]:
                mov2d = self.MOV_DOWN
                # if abs(deltz) < Z_THRESHOLD:
                #     return mov2d
                # else:
                #     return self.MOV_FRONT_DOWN if deltz > 0 else self.MOV_BACK_DOWN
            elif ang2d >= angrange[1] and ang2d < angrange[3]:
                mov2d = self.MOV_LEFT_DOWN if deltx > 0 else self.MOV_RIGHT_DOWN
                # if abs(deltz) < Z_THRESHOLD:
                #     return mov2d
                # else:
                #     if mov2d == self.MOV_LEFT_DOWN:
                #         return self.MOV_FRONT_LEFT_DOWN if deltz > 0 else self.MOV_BACK_RIGHT_DOWN
                #     else:
                #         return self.MOV_FRONT_RIGHT_DOWN if deltz > 0 else self.MOV_BACK_RIGHT_DOWN
            elif ang2d >= angrange[3] and ang2d < angrange[5]:
                mov2d = self.MOV_LEFT if deltx > 0 else self.MOV_RIGHT
                # if abs(deltz) < Z_THRESHOLD:
                #     return mov2d
                # else:
                #     if mov2d == self.MOV_LEFT:
                #         return self.MOV_FRONT_LEFT if deltz > 0 else self.MOV_BACK_LEFT
                #     else:
                #         return self.MOV_FRONT_RIGHT if deltz > 0 else self.MOV_BACK_RIGHT
            elif ang2d >= angrange[5] and ang2d < angrange[7]:
                mov2d = self.MOV_LEFT_UP if deltx > 0 else self.MOV_RIGHT_UP
                # if abs(deltz) < Z_THRESHOLD:
                #     return mov2d
                # else:
                #     if mov2d == self.MOV_LEFT_UP:
                #         return self.MOV_FRONT_LEFT_UP if deltz > 0 else self.MOV_BACK_LEFT_UP
                #     else:
                #         return self.MOV_FRONT_RIGHT_UP if deltz > 0 else self.MOV_BACK_RIGHT_UP
            else:
                mov2d = self.MOV_UP
                # if abs(deltz) < Z_THRESHOLD:
                #     return mov2d
                # else:
                #     return self.MOV_FRONT_UP if deltz > 0 else self.MOV_BACK_UP
            return mov2d
        else:
            # if abs(deltz) > Z_THRESHOLD:
            #     return self.MOV_FRONT if deltz > 0 else self.MOV_BACK
            # else:
            return self.MOV_STAY

    def __init__(self):
        super(DynamicRoutineGestureDetector, self).__init__()
        self.stay_count = 0
        self.training_count = 0
        # self.dynamic_gestures = {
        #     'gest': self.GEST_NONE,
        # }
        try:
            self.dynamic_gestures = pickle.load(open('dynamic_gesture.pkl', 'r'))
        except:
            self.dynamic_gestures = {
                'gest': self.GEST_NONE,
            }
        self.training_begin = False

    def detect(self):
        if self.stay_count > 5:
            print('Calculating')

            movements = []
            for ind in xrange(1, len(self.palm_positions)):
                mv = self.get_move(self.palm_positions[ind - 1], self.palm_positions[ind])
                if mv != self.MOV_STAY:
                    movements.append(mv)

            if not self.training_begin:
                self.training_begin = True
            else:
                if self.training_count > 10:
                    current_node = self.dynamic_gestures
                    current_node_name = None
                    for mv in movements:
                        if mv == current_node_name:
                            continue
                        if mv in current_node:
                            current_node = current_node[mv]
                            current_node_name = mv
                        else:
                            print('Unrecognized')
                            break
                    else:
                        print('Gest: %s' % current_node['gest'])
                else:
                    current_node = self.dynamic_gestures
                    current_node_name = None
                    for mv in movements:
                        if mv == current_node_name:
                            continue
                        if mv not in current_node:
                            current_node[mv] = {'gest': self.GEST_NONE}
                        current_node = current_node[mv]
                        current_node_name = mv
                    current_node['gest'] = self.GEST_7
                    self.training_count += 1

                    if self.training_count > 10:
                        pickle.dump(self.dynamic_gestures, open('dynamic_gesture.pkl', 'w+'))
                        print('Dump %s' % self.dynamic_gestures)
            self.clear()
            self.stay_count = 0

        if len(self.palm_positions) > 1:
            if self.get_move(self.palm_positions[-2], self.palm_positions[-1]) == self.MOV_STAY:
                self.stay_count += 1
                return
            else:
                self.stay_count = 0


class DynamicActionGestureDetector(GestureDetector):

    def __init__(self):
        super(DynamicActionGestureDetector, self).__init__()

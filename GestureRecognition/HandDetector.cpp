//
//  HandDetector.cpp
//  GestureRecognition
//
//  Created by ELTON on 19/3/14.
//  Copyright (c) 2014年 钟宇腾. All rights reserved.
//

#include "HandDetector.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/legacy/compat.hpp"

#include <vector>
using std::vector;

#include <cmath>
#include <algorithm>

#include <iostream>

using namespace cv;

bool HandDetector::grab(VideoCapture & capture) {
    return capture.grab()
        && capture.retrieve(depthDisparityImage, CAP_OPENNI_DISPARITY_MAP)
        && capture.retrieve(bgrImage, CAP_OPENNI_BGR_IMAGE)
        && capture.retrieve(validDepthMask, CAP_OPENNI_VALID_DEPTH_MASK);
}

void HandDetector::getHSVMask(Mat & hsvMask) const {
    Mat hsvImg(bgrImage.size(), CV_8UC3);
    GaussianBlur(bgrImage, hsvImg, Size(11, 11), 0);
    medianBlur(hsvImg, hsvImg, 11);
    
    cvtColor(hsvImg, hsvImg, COLOR_BGR2HSV);

    Mat hsvMask1, hsvMask2;
    inRange(hsvImg, Scalar(0, 30, 30), Scalar(40, 170, 255), hsvMask1);
    inRange(hsvImg, Scalar(156, 30, 30), Scalar(180, 170, 255), hsvMask2);
    bitwise_or(hsvMask1, hsvMask2, hsvMask);
    
    // Filtering
    Mat structuringElem = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(hsvMask, hsvMask, structuringElem);
    morphologyEx(hsvMask, hsvMask, MORPH_OPEN, structuringElem);
    dilate(hsvMask, hsvMask, structuringElem);
    morphologyEx(hsvMask, hsvMask, MORPH_CLOSE, structuringElem);
    
    GaussianBlur(hsvMask, hsvMask, Size(3, 3), 0);
}

void HandDetector::getFilteredDepthMap(Mat &maskedMap) const {
    Mat hsvMask, filteredDepthMap;
    getHSVMask(hsvMask);
    depthDisparityImage.copyTo(filteredDepthMap, hsvMask);
    
    Mat structuringElem = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(filteredDepthMap, filteredDepthMap, structuringElem);
    morphologyEx(filteredDepthMap, filteredDepthMap, MORPH_OPEN, structuringElem);
    dilate(filteredDepthMap, filteredDepthMap, structuringElem);
    morphologyEx(filteredDepthMap, filteredDepthMap, MORPH_CLOSE, structuringElem);
    
    medianBlur(filteredDepthMap, filteredDepthMap, 3);
    
    Mat depthMapMask;
    adaptiveThreshold(filteredDepthMap, depthMapMask, 255,
                      ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 0);
    filteredDepthMap.copyTo(maskedMap, depthMapMask);
}

vector<Point> HandDetector::getHandContour() const {
    Mat filteredDepthMap;
    getFilteredDepthMap(filteredDepthMap);

    vector<vector<Point>> contours;
    vector<Vec4i> hieachery;
    
    findContours(filteredDepthMap, contours, hieachery, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    vector<vector<Point>> filteredContours;
    using std::max_element;
    using std::remove_if;
    remove_copy_if(contours.begin(), contours.end(), std::back_inserter(filteredContours),
                   [](const vector<Point> & vec) {
        static const double MINIMUN_AREA = 5000.0;
        return fabs(contourArea(vec)) < MINIMUN_AREA;
    });
    
    auto max_iter = max_element(filteredContours.begin(), filteredContours.end(),
                                [&](const vector<Point> & a, const vector<Point> & b) {
                                        Rect roi_a(boundingRect(a)), roi_b(boundingRect(b));
                                        Mat crop_a(filteredDepthMap, roi_a), crop_b(filteredDepthMap, roi_b);
                                        Mat roiMask_a(Mat::zeros(crop_a.size(), CV_8UC1));
                                        Mat roiMask_b(Mat::zeros(crop_b.size(), CV_8UC1));
                                        drawContours(roiMask_a, vector<vector<Point>>(1, a), -1,
                                                     Scalar(255), CV_FILLED, CV_AA, noArray(),
                                                     1, -roi_a.tl());
                                        drawContours(roiMask_b, vector<vector<Point>>(1, b), -1,
                                                     Scalar(255), CV_FILLED, CV_AA, noArray(),
                                                     1, -roi_b.tl());
                                        Scalar _mean_a = mean(crop_a, roiMask_a);
                                        Scalar _mean_b = mean(crop_b, roiMask_b);
                                        Scalar _sum_a = sum(_mean_a);
                                        Scalar _sum_b = sum(_mean_b);

                                        return _sum_a[0] < _sum_b[0];
                                    
                                    });
    if (max_iter == filteredContours.end()) {
        return {};
    }
    else {
        return std::move(*max_iter);
    }
}

const Mat & HandDetector::getDepthDisparityImage() const {
    return depthDisparityImage;
}

const Mat & HandDetector::getBGRImage() const {
    return bgrImage;
}

vector<Point> HandDetector::getApproxPoly(const vector<Point> & contour) {
    if (contour.empty()) return {};
    vector<Point> poly;
    approxPolyDP(contour, poly, 5, false);
    return poly;
}

vector<Point> HandDetector::getConvexHull(const vector<Point> & contour) {
    if (contour.empty()) return {};
    vector<Point> hull;
    convexHull(contour, hull);
    return hull;
}

vector<Point> HandDetector::getConcavePoints(const vector<Point> & contour) {
    if (contour.empty()) return {};
    vector<int> hull;
    convexHull(contour, hull, true);
    vector<Vec4i> hulldef;
    convexityDefects(contour, hull, hulldef);
    
    vector<Point> result;
    for (auto & item : hulldef) {
        Point vec1 = contour[item[0]] - contour[item[2]];
        Point vec2 = contour[item[1]] - contour[item[2]];
        
        int dot = vec1.dot(vec2);
        
        if (dot >= -0.5) {
            result.push_back(contour[item[2]]);
        }
    }
    return result;
}

Point HandDetector::getPolyCenter(const vector<Point> & poly) {
    Moments mom = moments(poly, true);
    
    Point center(mom.m10 / mom.m00, mom.m01 / mom.m00);
    return center;
}

std::tuple<vector<Point>, vector<Point>> HandDetector::getFingers(const vector<Point> & poly) {
    if (poly.empty()) return {};
    
    vector<Point> fingers, concaves;
    
    vector<int> hull;
    convexHull(poly, hull, true);
    vector<Vec4i> hulldef;
    convexityDefects(poly, hull, hulldef);

    for (auto & item : hulldef) {
        Point vec1 = poly[item[0]] - poly[item[2]];
        Point vec2 = poly[item[1]] - poly[item[2]];
        
        int dot = vec1.dot(vec2);
        
        if (dot >= -0.5) {
            if (!fingers.empty()) {
                int deltx = fingers.back().x - poly[item[0]].x;
                int delty = fingers.back().y - poly[item[0]].y;
                int delt_pow = deltx * deltx - delty * delty;
                
                if (delt_pow > 10 * 10) {
                    fingers.push_back(poly[item[0]]);
                } else {
                    fingers.back().x = (fingers.back().x + poly[item[0]].x) / 2;
                    fingers.back().y = (fingers.back().y + poly[item[0]].y) / 2;
                }
            }
            if (!concaves.empty()) {
                int deltx = concaves.back().x - poly[item[2]].x;
                int delty = concaves.back().y - poly[item[2]].y;
                int delt_pow = deltx * deltx - delty * delty;
                
                if (delt_pow > 10 * 10) {
                    concaves.push_back(poly[item[2]]);
                } else {
                    concaves.back() = Point((concaves.back().x + poly[item[2]].x) / 2,
                                            (concaves.back().y + poly[item[2]].y) / 2);
                }
            } else {
                concaves.push_back(poly[item[2]]);
            }
            fingers.push_back(poly[item[1]]);
        }
    }
    
    return std::make_tuple(fingers, concaves);
}
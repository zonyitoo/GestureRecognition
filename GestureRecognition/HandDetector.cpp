//
//  HandDetector.cpp
//  GestureRecognition
//
//  Created by ELTON on 19/3/14.
//  Copyright (c) 2014年 钟宇腾. All rights reserved.
//

#include "HandDetector.h"
#include "utils.h"

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
        && capture.retrieve(depthDisparityImage, CV_CAP_OPENNI_DISPARITY_MAP)
        && capture.retrieve(bgrImage, CV_CAP_OPENNI_BGR_IMAGE)
        && capture.retrieve(validDepthMask, CV_CAP_OPENNI_VALID_DEPTH_MASK);
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
    Mat structuringElem = getStructuringElement(MORPH_RECT, Size(5, 5));
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
    
    imshow("DEPTH", filteredDepthMap);
    
    Mat structuringElem = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(filteredDepthMap, filteredDepthMap, structuringElem);
    morphologyEx(filteredDepthMap, filteredDepthMap, MORPH_OPEN, structuringElem);
    dilate(filteredDepthMap, filteredDepthMap, structuringElem);
    morphologyEx(filteredDepthMap, filteredDepthMap, MORPH_CLOSE, structuringElem);
    
    //GaussianBlur(filteredDepthMap, filteredDepthMap, Size(3, 3), 1);
    medianBlur(filteredDepthMap, filteredDepthMap, 3);
    
    Mat depthMapMask;
    adaptiveThreshold(filteredDepthMap, depthMapMask, 255,
                      ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 0);
    filteredDepthMap.copyTo(maskedMap, depthMapMask);
}

vector<vector<Point>> HandDetector::getHandContour() const {
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
        return vector<vector<Point>>(1, std::move(*max_iter));
    }
    
//    return filteredContours;
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
    approxPolyDP(contour, poly, 8, true);
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
    
//    vector<int> hull;
//    convexHull(poly, hull, true);
//    vector<Vec4i> hulldef;
//    convexityDefects(poly, hull, hulldef);
//
//    for (auto & item : hulldef) {
//        Point vec1 = poly[item[0]] - poly[item[2]];
//        Point vec2 = poly[item[1]] - poly[item[2]];
//        
//        int dot = vec1.dot(vec2);
//        
//        if (dot >= -0.5) {
//            if (!fingers.empty()) {
//                int deltx = fingers.back().x - poly[item[0]].x;
//                int delty = fingers.back().y - poly[item[0]].y;
//                int delt_pow = deltx * deltx - delty * delty;
//                
//                if (delt_pow > 5 * 5) {
//                    fingers.push_back(poly[item[0]]);
//                }
//            }
//            if (!concaves.empty()) {
//                int deltx = concaves.back().x - poly[item[2]].x;
//                int delty = concaves.back().y - poly[item[2]].y;
//                int delt_pow = deltx * deltx - delty * delty;
//                
//                if (delt_pow > 4 * 4) {
//                    concaves.push_back(poly[item[2]]);
//                }
//            } else {
//                concaves.push_back(poly[item[2]]);
//            }
//            fingers.push_back(poly[item[1]]);
//        }
//    }
//    static const size_t L1 = 8;
//    static const size_t L2 = 8;
    static const size_t L1 = 1;
    static const size_t L2 = 1;
    static const long double PI = acos(-1);
    static const long double cos_T = cos(95.0l / 180.0l * PI);
    vector<std::tuple<Point, long double, double>> filtered_pnts;
    for (long i = 0; i < poly.size(); ++ i) {
        long double greatest_cos = -1;
        double cross = 0.0;
        for (long leftind = i - L1; leftind < i; ++ leftind) {
            size_t j = (leftind < 0) ? (poly.size() - 1 + leftind) : leftind;
            for (long rightind = i + 1; rightind <= i + L2; ++ rightind) {
                size_t k = (rightind >= poly.size()) ? (rightind - poly.size()) : rightind;
                int deltx, delty;
                Point vec1 = poly[j] - poly[i];
                deltx = poly[j].x - poly[i].x;
                delty = poly[j].y - poly[i].y;
                long double dist1 = sqrt(deltx * deltx + delty * delty);
                Point vec2 = poly[k] - poly[i];
                deltx = poly[k].x - poly[i].x;
                delty = poly[k].y - poly[i].y;
                long double dist2 = sqrt(deltx * deltx + delty * delty);
                long double cosval = vec1.dot(vec2) / (dist1 * dist2);
                if (greatest_cos < cosval) {
                    greatest_cos = cosval;
                    cross = vec1.cross(vec2);
                }
            }
        }
        
        if (greatest_cos > cos_T || fabs(greatest_cos - cos_T) < 1e-8) {
            filtered_pnts.push_back(std::make_tuple(poly[i], greatest_cos, cross));
        }
    }
    
    for (const auto & tup : filtered_pnts) {
        Point p;
        double dot, cros;
        std::tie(p, dot, cros) = tup;
        
        if (cros > 0) {
            fingers.push_back(p);
        } else {
            concaves.push_back(p);
        }
    }
    
//    if (fingers.size() > 7)
//        fingers.resize(7);
//    if (concaves.size() > 7)
//        concaves.resize(7);
    
    return std::make_tuple(fingers, concaves);
}

std::tuple<vector<Point>, vector<Point>> HandDetector::getKmeanFingers(const vector<Point> & poly) {
    static const size_t L1 = 5;
    static const size_t L2 = 5;
    static const long double PI = acos(-1);
    static const long double cos_T = cos(95.0l / 180.0l * PI);
    //vector<std::tuple<Point, long double, double>> filtered_pnts;
    vector<Point> filtered_pnts;
    for (long i = 0; i < poly.size(); ++ i) {
        long double greatest_cos = -1;
        double cross = 0.0;
        for (long leftind = i - L1; leftind < i; ++ leftind) {
            size_t j = (leftind < 0) ? (poly.size() - 1 + leftind) : leftind;
            for (long rightind = i + 1; rightind <= i + L2; ++ rightind) {
                size_t k = (rightind >= poly.size()) ? (rightind - poly.size()) : rightind;
                int deltx, delty;
                Point vec1 = poly[j] - poly[i];
                deltx = poly[j].x - poly[i].x;
                delty = poly[j].y - poly[i].y;
                long double dist1 = sqrt(deltx * deltx + delty * delty);
                Point vec2 = poly[k] - poly[i];
                deltx = poly[k].x - poly[i].x;
                delty = poly[k].y - poly[i].y;
                long double dist2 = sqrt(deltx * deltx + delty * delty);
                long double cosval = vec1.dot(vec2) / (dist1 * dist2);
                if (greatest_cos < cosval) {
                    greatest_cos = cosval;
                    cross = vec1.cross(vec2);
                }
            }
        }
        
        if ((greatest_cos > cos_T || fabs(greatest_cos - cos_T) < 1e-8) && cross > 0) {
            //filtered_pnts.push_back(std::make_tuple(poly[i], greatest_cos, cross));
            filtered_pnts.push_back(poly[i]);
        }
    }
    
    vector<Point> kmean_centers = kmeans_cluster(filtered_pnts, 10);
    
    Point center = getPolyCenter(kmean_centers);
    std::sort(kmean_centers.begin(), kmean_centers.end(),
              [&center] (const std::tuple<Point, long double, double> & a, const std::tuple<Point, long double, double> & b) {
                  Point vec1 = std::get<0>(a) - center;
                  Point vec2 = std::get<0>(b) - center;
                  
                  double atan_val1 = atan2(vec1.x, vec1.y);//(vec1.x != 0) ? atan2(vec1.y, vec1.x) : (vec1.y < 0) ? -DBL_MAX : DBL_MAX;
                  double atan_val2 = atan2(vec2.x, vec2.y);//(vec2.x != 0) ? atan2(vec2.y, vec2.x) : (vec2.y < 0) ? -DBL_MAX : DBL_MAX;
                  return atan_val1 < atan_val2;
              });
    
    vector<Point> fingers, concaves;
    fingers = kmean_centers;
    return std::make_tuple(std::move(fingers), std::move(concaves));
}
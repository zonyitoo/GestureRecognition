//
//  main.cpp
//  GestureRecognition
//
//  Created by ELTON on 19/3/14.
//  Copyright (c) 2014年 钟宇腾. All rights reserved.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/legacy/compat.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>

#include "HandDetector.h"

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::get;

static void colorizeDisparity(Mat &disparityMap, Mat &colorizedMap) {
    colorizedMap.create(disparityMap.size(), CV_8UC3);
    colorizedMap = Scalar::all(0);
    
    for (int row = 0; row < disparityMap.rows; ++ row) {
        for (int col = 0; col < disparityMap.cols; ++ col) {
            unsigned char d = disparityMap.at<unsigned char>(row, col);
            
            unsigned char r = 0, g = 0, b = 0;
            if (d % 9 < 3) {
                r = 100;
            } else if (d % 9 < 6) {
                g = 100;
            } else {
                b = 100;
            }
            
            colorizedMap.at<Point3_<unsigned char>>(row, col) = Point3_<unsigned char>(r, g, b);
        }
    }
}

static void findHand(const Mat &depthMap, const Mat &src, Mat &hand) {
    Mat hsvImg(src.size(), CV_8UC3);
    GaussianBlur(src, hsvImg, Size(11, 11), 0);
    medianBlur(hsvImg, hsvImg, 11);
    
    cvtColor(hsvImg, hsvImg, COLOR_BGR2HSV);
    imshow("HSV IMG", hsvImg);
    
    Mat hsvMask1, hsvMask2;
    inRange(hsvImg, Scalar(0, 30, 30), Scalar(40, 170, 255), hsvMask1);
    inRange(hsvImg, Scalar(156, 30, 30), Scalar(180, 170, 255), hsvMask2);
    Mat hsvMask;
    bitwise_or(hsvMask1, hsvMask2, hsvMask);
    
    // Filtering
    Mat structuringElem = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(hsvMask, hsvMask, structuringElem);
    morphologyEx(hsvMask, hsvMask, MORPH_OPEN, structuringElem);
    dilate(hsvMask, hsvMask, structuringElem);
    morphologyEx(hsvMask, hsvMask, MORPH_CLOSE, structuringElem);
    
    GaussianBlur(hsvMask, hsvMask, Size(3, 3), 0);
    imshow("HSV MASK", hsvMask);
    
    Mat filteredDepthMap;
    depthMap.copyTo(filteredDepthMap, hsvMask);
    
    erode(filteredDepthMap, filteredDepthMap, structuringElem);
    morphologyEx(filteredDepthMap, filteredDepthMap, MORPH_OPEN, structuringElem);
    dilate(filteredDepthMap, filteredDepthMap, structuringElem);
    morphologyEx(filteredDepthMap, filteredDepthMap, MORPH_CLOSE, structuringElem);
    
    medianBlur(filteredDepthMap, filteredDepthMap, 5);
    
    Mat depthMapMask;
    adaptiveThreshold(filteredDepthMap, depthMapMask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 0);
    //threshold(filteredDepthMap, depthMapMask, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Depth Mask", depthMapMask);
    Mat masked;
    filteredDepthMap.copyTo(masked, depthMapMask);
    //threshold(filteredDepthMap, filteredDepthMap, 0, 255, THRESH_TOZERO | THRESH_OTSU);
    imshow("Depth Map", masked);
    
    vector<vector<Point>> contours;
    //vector<vector<Point>> filtedContours;
    vector<vector<Point>> contourHulls;
    vector<Vec4i> hieachery;
    
    findContours(masked, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    drawContours(hand, contours, -1, Scalar(255));
    
    vector<std::pair<double, double>> sumDepth;
    vector<vector<Point>> fixedContours;
    vector<vector<Vec4i>> defects;
    int max_index = -1;
    double max_sum = 0.0;
    for (int i = 0; i < contours.size(); ++ i) {
        double area = fabs(contourArea(Mat(contours[i])));
        if (area > 1000) {
            Rect roi(boundingRect(contours[i]));
            Mat crop(depthMap, roi);
            Mat roiMask(Mat::zeros(crop.size(), CV_8UC1));
            drawContours(roiMask, contours, i, Scalar(255), CV_FILLED, CV_AA, noArray(), 1, -roi.tl());
            Scalar _mean = mean(crop, roiMask);
            Scalar _sum = sum(_mean);
            
            //sumDepth.push_back(std::make_pair(_sum[0], area));
            
            vector<Point> poly;
            approxPolyDP(contours[i], poly, 5, false);
            fixedContours.push_back(poly);
            if (max_sum < _sum[0]) {
                max_sum = _sum[0];
                max_index = static_cast<int>(fixedContours.size()) - 1;
            }
            
            vector<Point> h;
            //convexHull(contours[i], h, true);
            convexHull(poly, h, true);
            contourHulls.push_back(h);
            
            vector<Vec4i> def;
            vector<int> ind_hull;
            //convexHull(contours[i], ind_hull, true);
            //convexityDefects(contours[i], ind_hull, def);
            convexHull(poly, ind_hull, true);
            convexityDefects(poly, ind_hull, def);
            defects.push_back(def);
        }
    }
    
    if (max_index != -1) {
        drawContours(hand, fixedContours, max_index, Scalar(0, 255));
        drawContours(hand, contourHulls, max_index, Scalar(0, 0, 255));
        
        //        vector<Point> poly;
        //        approxPolyDP(fixedContours[max_index], poly, 10, true);
        //        drawContours(hand, vector<vector<Point>>(1, poly), -1, Scalar(123, 123, 12));
        
        vector<Point> fingers;
//        fingers.push_back(contourHulls[max_index][0]);
//        
//        for (int i = 1; i < contourHulls[max_index].size(); ++ i) {
//            auto &vec = contourHulls[max_index];
//            if (fabs(fingers.back().x - vec[i].x) < 3.0f || fabs(fingers.back().y - vec[i].y) < 3.0f) {
//                fingers.back() = Point((fingers.back().x + vec[i].x) / 2.0, (fingers.back().y + vec[i].y) / 2.0);
//            } else {
//                fingers.push_back(vec[i]);
//            }
//        }
        //        vector<Point> fingers;
        //        for (auto &pnt : poly) {
        //            fingers.push_back(pnt);
        //        }
        
//        sort(fingers.begin(), fingers.end(), [](const Point & p1, const Point & p2) {
//            return p1.y < p2.y;
//        });
//        fingers.resize(5);
//        
//        for (auto &pnt : fingers) {
//            circle(hand, pnt, 5, Scalar(255, 255, 0), 1);
//        }
        
        for (auto & item : defects[max_index]) {
            if (item[3] / 255 > 5) {
                circle(hand, fixedContours[max_index][item[2]], 5, Scalar(255, 100, 80), 5);
            }
        }
    }
    /*
     int biggest_index = 0;
     float biggest_area = 0.0f;
     for (int i = 0; i < contours.size(); ++ i) {
     auto m = Mat(contours[i]);
     float f = fabs(contourArea(m));
     if (f > 14000 && biggest_area < f) {
     //filtedContours.push_back(itr);
     biggest_area = f;
     biggest_index = i;
     }
     
     vector<Point> h;
     convexHull(contours[i], h, true);
     contourHulls.push_back(h);
     }
     
     drawContours(hand, contours, biggest_index, Scalar(255));
     drawContours(hand, contourHulls, biggest_index, Scalar(0, 255));*/
    
    //drawContours(hand, filtedContours, -1, Scalar(255));
    
//    vector<Point> hull;
//    convexHull(contours[biggest_index], hull, true);
//    
//    vector<Vec4i> conDefs;
//    convexityDefects(contours[biggest_index], hull, conDefs);
    
}

int main(int argc, char *argv[]) {
    
    std::ios_base::sync_with_stdio(false);
    
    VideoCapture capture;
    capture.open(CAP_OPENNI);
    
    if (!capture.isOpened()) {
        cerr << "Cannot open video capture" << endl;
        return EXIT_FAILURE;
    }
    
    HandDetector dec;
    
    while (dec.grab(capture)) {
        vector<Point> handContour = dec.getHandContour();

        Mat hand(dec.getBGRImage());
        
        vector<Point> poly = dec.getApproxPoly(handContour);
        drawContours(hand, vector<vector<Point>>(1, poly), -1, Scalar(255, 255));
        
        vector<Point> hull = dec.getConvexHull(poly);
        drawContours(hand, vector<vector<Point>>(1, hull), -1, Scalar(0, 0, 255));
        
        Point palm_center = HandDetector::getPolyCenter(poly);
        circle(hand, palm_center, 10, Scalar(0, 255, 255), 5);
        
        vector<Point> fingers, conv;
        std::tie(fingers, conv) = dec.getFingers(poly);
        for (auto & pnt : fingers) {
            circle(hand, pnt, 5, Scalar(0, 200, 100), 5);
        }
        
        for (auto & pnt : conv) {
            circle(hand, pnt, 5, Scalar(0, 255), 5);
        }
        
        if (poly.empty()) continue;
        RotatedRect elip = fitEllipse(poly);
        ellipse(hand, elip, Scalar(200, 10, 123));
        
        imshow("HAND", hand);
        
        if (waitKey(60) == 27) {
            break;
        }
    }
    
    //while (capture.grab()) {
        /*
         Mat depthMap;
         if (capture.retrieve(depthMap, CAP_OPENNI_DEPTH_MAP)) {
         Mat showDepthMap;
         depthMap.convertTo(showDepthMap, CV_8UC1, 0.05f);
         imshow("DepthMap", depthMap);
         imshow("Show", showDepthMap);
         }
         Mat disparityMap;
         if (capture.retrieve(disparityMap, CAP_OPENNI_DISPARITY_MAP)) {
         imshow("Original disparityMap", disparityMap);
         
         Mat colorizedMap;
         colorizeDisparity(disparityMap, colorizedMap);
         imshow("Colorized disparityMap", colorizedMap);
         
         Mat hand;
         hand.create(disparityMap.size(), CV_8UC3);
         hand = Scalar::all(0);
         for (int row = 0; row < disparityMap.rows; ++ row) {
         for (int col = 0; col < disparityMap.cols; ++ col) {
         auto d = disparityMap.at<unsigned char>(row, col);
         
         if (d > 60 && d < 200) {
         hand.at<Point3_<unsigned char>>(row, col) = Point3_<unsigned char>(d + 10, d + 60, 0);
         } else {
         hand.at<Point3_<unsigned char>>(row, col) = Point3_<unsigned char>();
         }
         }
         }
         imshow("Hand", hand);
         }
         
         Mat validDepthMap;
         if (capture.retrieve(validDepthMap, CAP_OPENNI_VALID_DEPTH_MASK)) {
         imshow("valid depth mask", validDepthMap);
         }*/
        
//        Mat bgrMap, depthMap, validDepthMask;
//        if (capture.retrieve(bgrMap, CAP_OPENNI_BGR_IMAGE)
//                && capture.retrieve(depthMap, CAP_OPENNI_DISPARITY_MAP)
//                && capture.retrieve(validDepthMask, CAP_OPENNI_VALID_DEPTH_MASK)) {
//            Mat hand, bgr;
//            bgrMap.copyTo(bgr, validDepthMask);
//            hand.create(bgr.size(), CV_8UC3);
//            findHand(depthMap, bgrMap, hand);
//            imshow("Hand", hand);
//        }
//        
//        if (waitKey(30) == 27) {
//            break;
//        }
//    }
    
    return 0;
}


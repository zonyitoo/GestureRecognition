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


int main(int argc, char *argv[]) {
    
    std::ios_base::sync_with_stdio(false);
    
    VideoCapture capture;
    capture.open(CV_CAP_OPENNI);
    
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


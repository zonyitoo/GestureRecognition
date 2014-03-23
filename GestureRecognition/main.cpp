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
#include <cstdlib>
#include <unistd.h>

#include "HandDetector.h"
#include "utils.h"

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::get;

inline long dist_2D(const Point & a, const Point & b) {
    long deltx = a.x - b.x;
    long delty = a.y - b.y;
    return deltx * deltx + delty * delty;
}

int main(int argc, char *argv[]) {
    
    std::ios_base::sync_with_stdio(false);
    
    VideoCapture capture;
    capture.open(CV_CAP_OPENNI);
    
    if (!capture.isOpened()) {
        cerr << "Cannot open video capture" << endl;
        return EXIT_FAILURE;
    }
    
    HandDetector dec;
    
    vector<vector<Point>> routine;
    
    const Scalar ROUTINE_COLORS[5] = {
        {0, 255, 0},
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 255},
        {0, 255, 255}
    };
    
    while (dec.grab(capture)) {
        Mat hand(dec.getBGRImage()), distImg(dec.getDepthDisparityImage().size(), CV_8UC3);
        auto handContour = dec.getHandContour();
        
//        vector<Point> fingers, conv;
//        std::tie(fingers, conv) = dec.getFingers(handContour);
        
        //vector<Point> clus = kmeans_cluster(fingers, 15);
        
//        for (auto & cluspnt : clus) {
//            circle(hand, cluspnt, 4, Scalar(255), 4);
//        }

        drawContours(hand, handContour, -1, Scalar(0, 255));
        for (auto & cont : handContour) {
            Mat depth8u(Mat::zeros(dec.getBGRImage().size(), CV_8UC1));
            drawContours(depth8u, vector<vector<Point>>(1, cont), -1, Scalar(255), CV_FILLED);
            imshow("DEPTH8U", depth8u);
            RotatedRect minRect = minAreaRect(cont);
            Point2f pt[4];
            minRect.points(pt);
            for (int i = 0; i < 4; ++ i) {
                line(hand, pt[i], pt[(i + 1) % 4], Scalar(255));
            }
            
            Mat outp(Mat::zeros(depth8u.size(), CV_32FC1));
            distanceTransform(depth8u, outp, CV_DIST_L2, CV_DIST_MASK_5);
            float maxdist = 0.0;
            float mindist = std::numeric_limits<float>().max();
            Point max_pnt;
            for (int i = 0; i < outp.rows; ++ i) {
                for (int j = 0; j < outp.cols; ++ j) {
                    auto d = outp.at<float>(i, j);
                    if (maxdist < d) {
                        maxdist = d;
                        max_pnt = Point(j, i);
                    }
                    
                    if (d != 0 && d > mindist) {
                        mindist = d;
                    }
                }
            }
            
            circle(hand, max_pnt, 5, Scalar(0, 255,255), 5);
            
            //RotatedRect palm_area(max_pnt, Size2f(maxdist * 2, maxdist * 2), minRect.angle);
//            Rect palm_area(Point(max_pnt.y - maxdist, max_pnt.x - maxdist), Size(maxdist * 2, maxdist * 2));
//            Mat filt, filt_mask = Mat::zeros(outp.size(), CV_8UC1);
//            for (int i = 0; i < outp.rows; ++ i) {
//                for (int j = 0; j < outp.cols; ++ j) {
//                    auto d = outp.at<float>(i, j);
////                    if (d > 0.5 && d <= maxdist / 4) {
////                        filt_mask.at<uchar>(i, j) = 255;
////                    }
//                    if (d != 0 && !Point(i, j).inside(palm_area)) {//dist_2D(Point(j, i), max_pnt) > maxdist * maxdist) {
//                        filt_mask.at<uchar>(i, j) = 255;
//                    }
//                }
//            }
            
            Mat filt_mask = Mat::zeros(outp.size(), CV_8U), tmp;
            outp.copyTo(tmp);
            Mat handstruc = getStructuringElement(MORPH_RECT, Size(maxdist / 3.5, maxdist / 3.5));
            erode(tmp, tmp, handstruc);
            //erode(filt, filt, struc);
            //morphologyEx(filt, filt, MORPH_OPEN, struc);
            //dilate(filt, filt, struc);
            morphologyEx(tmp, tmp, MORPH_OPEN, handstruc);
            morphologyEx(tmp, tmp, MORPH_CLOSE, handstruc);
            imshow("TMP", tmp);
            for (int row = 0; row < tmp.rows; ++ row) {
                for (int col = 0; col < tmp.cols; ++ col) {
                    float d = tmp.at<float>(row, col);
                    filt_mask.at<uchar>(row, col) = (d == 0) ? 255 : 0;
                }
            }
            imshow("FILT_MASK", filt_mask);
            
            
            Mat struc = getStructuringElement(MORPH_RECT, Size(maxdist / 4, maxdist / 4));
            Mat filt;
            outp.copyTo(filt, filt_mask);
            imshow("FILT", filt);
            morphologyEx(filt, filt, MORPH_OPEN, struc);
            //morphologyEx(filt, filt, MORPH_CLOSE, struc);
            filt.convertTo(distImg, CV_8UC3, 100);//, 255 / maxdist * 8);
            
            vector<vector<Point>> fingerContours;
            vector<Vec4i> hera;
            Mat filtc1;
            filt.convertTo(filtc1, CV_8UC1);
            findContours(filtc1, fingerContours, hera, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            drawContours(hand, fingerContours, -1, Scalar(0, 0, 255));
            for (auto & fingerCont : fingerContours) {
                
            }
// HoughLines
//            vector<Vec4i> lines;
//            Mat binfilt;
//            filt.convertTo(binfilt, CV_8UC1, 100);
//            imshow("BINFILT", binfilt);
//            HoughLinesP(binfilt, lines, 1, CV_PI / 180, 60, 30, 3);
//            
//            for (auto & vec : lines) {
//                line(hand, Point(vec[0], vec[1]), Point(vec[2], vec[3]), Scalar(0, 0, 255), 3, 8);
//            }
            
//            for (int row = 0; row < distImg.rows; ++ row) {
//                for (int col = 0; col < distImg.cols; ++ col) {
//                    if (distImg.at<uchar>(row, col) != 0)
//                        circle(hand, Point(col, row), 2, Scalar(255));
//                }
//            }
//
//            Mat rotate8u(minRect.boundingRect().size(), CV_8UC1);
//            imshow("Before Rotate", rotate8u);
//            Mat after_rotate8u(minRect.boundingRect().size(), CV_8UC1);
//            getRectSubPix(depth8u, minRect.boundingRect().size(), minRect.center, rotate8u);
//            Point2f rotateCenter(minRect.boundingRect().size().width / 2.0, minRect.boundingRect().size().height / 2);
//            Mat rotateMatrix(getRotationMatrix2D(rotateCenter, 90 + minRect.angle, 1));
//            warpAffine(rotate8u, after_rotate8u, rotateMatrix, minRect.boundingRect().size());
//            Mat scale8u(Size(30, minRect.size.width), CV_8UC1);
//            getRectSubPix(after_rotate8u, Size(minRect.size.height, minRect.size.width), rotateCenter, scale8u);
//            Mat outputdepth32u(scale8u.size(), CV_32FC1);
//            distanceTransform(scale8u, outputdepth32u, CV_DIST_L2, CV_DIST_MASK_5);
//            imshow("AFTER", outputdepth32u);
            
//            vector<Point> rawpoly = dec.getApproxPoly(cont);
//            if (rawpoly.empty()) {
//                if (waitKey(60) == 27) {
//                    break;
//                }
//                continue;
//            }
//            
//            vector<Point> poly {rawpoly[0]};
//            
//            for (auto itr = rawpoly.begin() + 1; itr != rawpoly.end(); ++ itr) {
//                int deltx = itr->x - (itr - 1)->x;
//                int delty = itr->y - (itr - 1)->y;
//                int d = deltx * deltx + delty * delty;
//                
//                if (d >= 500) {
//                    poly.push_back(*itr);
//                }
//            }
//            
//            if (poly.size() >= 2) {
//                int deltx = poly.begin()->x - poly.rbegin()->x;
//                int delty = poly.begin()->y - poly.rbegin()->y;
//                int d = deltx * deltx + delty * delty;
//                if (d < 500) {
//                    *poly.begin() = Point((poly.begin()->x + poly.rbegin()->x) / 2,
//                                          (poly.begin()->y + poly.rbegin()->y) / 2);
//                    poly.pop_back();
//                }
//            }
//            
//            for (int i = 0; i < poly.size(); ++ i) {
//                circle(hand, poly[i], 2, Scalar(100 + i * 30), 2);
//            }
//            
//            //vector<Point> hull = dec.getConvexHull(poly);
//            //drawContours(hand, vector<vector<Point>>(1, hull), -1, Scalar(0, 0, 255));
//            
//            Point palm_center = HandDetector::getPolyCenter(poly);
//            
//            vector<Point> fingers, conv;
//            std::tie(fingers, conv) = dec.getFingers(poly);
//            
//            if (fingers.size() > 7) continue;
//            
//            //std::tie(fingers, conv) = dec.getKmeanFingers(handContour);
//            
//            circle(hand, palm_center, 10, Scalar(0, 255, 255), 5);
//            drawContours(hand, vector<vector<Point>>(1, poly), -1, Scalar(255, 255));
//            drawContours(hand, vector<vector<Point>>(1, fingers), -1, Scalar(255, 255));
//            
//            int _x = 0;
//            for (auto & pnt : fingers) {
//                circle(hand, pnt, 2, Scalar(0, 150 + 40 * _x ++, 100), 2);
//            }
//            
//            for (auto & pnt : conv) {
//                circle(hand, pnt, 2, Scalar(0, 0, 255), 2);
//            }
//            
//            RotatedRect rect = minAreaRect(poly);
//            //rectangle(hand, rect, Scalar(255, 0, 255));
//            Point2f rect_pnts[4];
//            rect.points(rect_pnts);
//            for (int i = 0; i < 4; ++ i) {
//                line(hand, rect_pnts[i], rect_pnts[(i + 1) % 4], Scalar(255, 0, 255));
//            }
//        }
        
//        static const long DIST_THRESHOLE_U = 100000l;
//        static const long DIST_THRESHOLE_L = 1000l;
//        
//        static const auto pnt_dist = [] (const Point & a, const Point & b) {
//            int deltx = a.x - b.x;
//            int delty = a.y - b.y;
//            return deltx * deltx + delty * delty;
//        };
//        
//        static int count = 0;
//        
//        bool can_add = true;
//        if (routine.size() > 1) {
//            for (int i = 0; i < fingers.size(); ++ i) {
//                if (fingers[i].x == 0 && fingers[i].y == 0) break;
//                
//                auto dist = pnt_dist(fingers[i], routine.back()[i]);
//                
//                if (dist > DIST_THRESHOLE_U || dist < DIST_THRESHOLE_L) {
//                    can_add = false;
//                    break;
//                }
//            }
//        }
//        
//        if (can_add) {
//            routine.push_back(fingers);
//            count = 0;
//        } else {
//            count ++;
//            
//            if (count > 30) {
//                routine.clear();
//                count = 0;
//            }
//        }

//        if (routine.size() > 1) {
//            for (int i = 1; i < routine.size(); ++ i) {
//                for (int j = 0; j < routine[i].size(); ++ j) {
//                    line(hand, routine[i - 1][j], routine[i][j], ROUTINE_COLORS[j]);
//                }
//            }
        }
        
        imshow("HAND", hand);
        imshow("DIST", distImg);
        
        if (waitKey(330) == 27) {
            break;
        }
    }
    
    capture.release();

    return 0;
}


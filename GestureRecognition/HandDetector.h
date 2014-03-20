//
//  HandDetector.h
//  GestureRecognition
//
//  Created by ELTON on 19/3/14.
//  Copyright (c) 2014年 钟宇腾. All rights reserved.
//

#ifndef __GestureRecognition__HandDetector__
#define __GestureRecognition__HandDetector__

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <tuple>

class HandDetector final {
public:
    bool grab(cv::VideoCapture & capture);
    std::vector<cv::Point> getHandContour() const;
    
    const cv::Mat & getDepthDisparityImage() const;
    const cv::Mat & getBGRImage() const;
    
    static std::vector<cv::Point> getApproxPoly(const std::vector<cv::Point> & contour);
    static std::vector<cv::Point> getConvexHull(const std::vector<cv::Point> & contour);
    static std::vector<cv::Point> getConcavePoints(const std::vector<cv::Point> & contour);
    static cv::Point getPolyCenter(const std::vector<cv::Point> & poly);
    static std::tuple<std::vector<cv::Point>, std::vector<cv::Point>> getFingers(const std::vector<cv::Point> & poly);
private:
    void getHSVMask(cv::Mat & hsvMask) const;
    void getFilteredDepthMap(cv::Mat & filteredDepthMap) const;
    
    cv::Mat depthDisparityImage;
    cv::Mat bgrImage;
    cv::Mat validDepthMask;
    
    cv::Mat hand;
};

#endif /* defined(__GestureRecognition__HandDetector__) */

//
//  utils.hpp
//  GestureRecognition
//
//  Created by ELTON on 21/3/14.
//  Copyright (c) 2014年 钟宇腾. All rights reserved.
//

#ifndef GestureRecognition_utils_hpp
#define GestureRecognition_utils_hpp

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::vector<cv::Point> kmeans_cluster(const std::vector<cv::Point> poly, unsigned int k);


#endif

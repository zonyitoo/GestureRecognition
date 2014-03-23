//
//  utils.cpp
//  GestureRecognition
//
//  Created by ELTON on 21/3/14.
//  Copyright (c) 2014年 钟宇腾. All rights reserved.
//

#include "utils.h"
#include <tuple>
#include <vector>
#include <iostream>

#include <climits>
#include <cstdlib>

#include <opencv2/core/core.hpp>

using std::tuple;
using std::vector;
using std::make_tuple;

struct KmeansPoint {
    int x, y;
    size_t group;
    
    KmeansPoint(): x(0), y(0), group(0) {}
    KmeansPoint(int x, int y, int group): x(x), y(y), group(group) {}
};

inline double randf(double m) {
    return m * rand() / (RAND_MAX - 1.);
}

inline long distance_2D(const KmeansPoint & a, const KmeansPoint & b) {
    long distx = a.x - b.x;
    long disty = a.y - b.y;
    return distx * distx + disty * disty;
}

std::tuple<size_t, long>
    kmeans_nearest_cluster_center(const KmeansPoint & point, const vector<KmeansPoint> & cluster_centers, size_t ncluster) {
    size_t min_index = point.group;
    long min_dist = LONG_MAX;
    
    for (size_t i = 0; i < ncluster; ++ i) {
        long d = distance_2D(cluster_centers[i], point);
        if (min_dist > d) {
            min_dist = d;
            min_index = i;
        }
    }
    
    return make_tuple(min_index, min_dist);
}

void kpp(vector<KmeansPoint> & points, vector<KmeansPoint> & cluster_centers) {
    cluster_centers[0] = points[rand() % points.size()];
    vector<long> d(points.size());
    
    // Find seeds
    for (auto itr = cluster_centers.begin() + 1; itr != cluster_centers.end(); ++ itr) {
        long sum = 0l;
        for (int j = 0; j < points.size(); ++ j) {
            d[j] = std::get<1>(kmeans_nearest_cluster_center(points[j], cluster_centers, itr - cluster_centers.begin()));
            sum += d[j];
        }
        
        sum = randf(sum);
        
        for (size_t j = 0; j < d.size(); ++ j) {
            sum -= d[j];
            if (sum > 0) continue;
            
            *itr = points[j];
            break;
        }
    }
    
    for (auto & pnt : points) {
        auto ret = kmeans_nearest_cluster_center(pnt, cluster_centers, cluster_centers.size());
        pnt.group = std::get<0>(ret);
    }
}

vector<KmeansPoint> lloyd(vector<KmeansPoint> & points, unsigned int nclusters) {
    vector<KmeansPoint> cluster_centers(nclusters);
    
    kpp(points, cluster_centers); // Init
    
    size_t lenpts10 = points.size() >> 10;
    
    while (true) {
        for (auto & cc : cluster_centers) {
            cc = KmeansPoint();
        }
        
        for (auto & p : points) {
            cluster_centers[p.group].group ++;
            cluster_centers[p.group].x += p.x;
            cluster_centers[p.group].y += p.y;
        }
        
        for (auto & cc : cluster_centers) {
            cc.x /= cc.group;
            cc.y /= cc.group;
        }
        
        size_t changed = 0;
        for (auto & p : points) {
            auto min_i = std::get<0>(kmeans_nearest_cluster_center(p, cluster_centers, cluster_centers.size()));
            if (min_i != p.group) {
                changed += 1;
                p.group = min_i;
                
            }
        }
        
        if (changed <= lenpts10) {
            break;
        }
    }
    
    for (size_t i = 0; i < cluster_centers.size(); ++ i) {
        cluster_centers[i].group = i;
    }
    
    return cluster_centers;
}

std::vector<cv::Point> kmeans_cluster(const std::vector<cv::Point> poly, unsigned int k) {
    vector<cv::Point> result;
    
    if (poly.size() <= k) return result;
    
    vector<KmeansPoint> points;
    for (auto & pnt : poly) {
        KmeansPoint kp;
        kp.x = pnt.x;
        kp.y = pnt.y;
        points.push_back(kp);
    }
    vector<KmeansPoint> clust_cnts = lloyd(points, k);
    
    for (auto & pnt : clust_cnts) {
        result.push_back(cv::Point(pnt.x, pnt.y));
    }
    
    return result;
}

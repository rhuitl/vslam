/*
 *  main.cpp
 *  outlet_detection
 *
 *  Created by Victor  Eruhimov on 4/16/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */


#include "posest/planarSFM.h"
#include <iostream>
#include <stdio.h>
#include <posest/test/simulated.h>

using namespace cv;

namespace pe
{

void test()
{
    vector<Point2f> points1, points2;

    points1.resize(6);
    points2.resize(6);

    points1[0] = Point2f(0, 0);
    points1[1] = Point2f(0, 1);
    points1[2] = Point2f(1, 0);
    points1[3] = Point2f(1, 1);
    points1[4] = Point2f(3, 0);
    points1[5] = Point2f(3, 1);

    points2 = points1;

    Mat H = findHomography(Mat(points1), Mat(points2), CV_RANSAC);

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            printf("%f ", H.at<float>(i, j));

}

void generatePlanarObject(vector<Point3f>& points, Point3f N, float d)
{
    // choose normal
//    Point3f N(0, 0, 1);//(1.0f - 2*float(rand())/RAND_MAX, 1.0f - 2*float(rand())/RAND_MAX, float(rand())/RAND_MAX);
//    float d = 10.0f;

    // generate points such that N*p - d = 0
    Point3f N0 = N*(d/norm(N));
    for(size_t i = 0; i < points.size(); i++)
    {
        Point3f p(float(rand())/RAND_MAX, float(rand())/RAND_MAX, float(rand())/RAND_MAX);
        Point3f p_plane = crossProduct(p, N);
        points[i] = p_plane + N0;
    }
}

// generate points in a cube
void generate3DPointCloud(vector<Point3f>& points, Point3f pmin, Point3f pmax)
{
    const Point3f delta = pmax - pmin;
    for(size_t i = 0; i < points.size(); i++)
    {
        Point3f p(float(rand())/RAND_MAX, float(rand())/RAND_MAX, float(rand())/RAND_MAX);
        p.x *= delta.x;
        p.y *= delta.y;
        p.z *= delta.z;
        p = p + pmin;

        points[i] = p;
    }
}

void addPointNoise(vector<Point2f>& points, double sigma = 3.0)
{
    RNG rng(1);
    for(size_t i = 0; i < points.size(); i++)
    {
        points[i].x += 2*rand()*sigma/RAND_MAX - sigma;
        points[i].y += 2*rand()*sigma/RAND_MAX - sigma;
    }
}

void addLinkNoise(vector<int>& indices, double ratio = 0.05)
{
    const int count = int(indices.size()*ratio/2);
    for(size_t i = 0; i < count; i++)
    {
        int index1 = rand()%indices.size();
        int index2 = rand()%indices.size();
        std::swap(indices[index1], indices[index2]);
    }
}

void addLinkNoise(vector<Match>& indices, double ratio)
{
    const int count = int(indices.size()*ratio/2);
    for(size_t i = 0; i < count; i++)
    {
        int index1 = rand()%indices.size();
        int index2 = rand()%indices.size();
        std::swap(indices[index1].index2, indices[index2].index2);
    }
}

void generateIntrinsics(Mat& intrinsics)
{
  intrinsics = Mat::eye(3, 3, CV_32F);
  intrinsics.at<float>(0, 0) = 400.0;
  intrinsics.at<float>(1, 1) = 400.0;
  intrinsics.at<float>(0, 2) = 640/2;
  intrinsics.at<float>(1, 2) = 480/2;
}

void generateData(Mat& intrinsics, Mat& R, Mat& T, vector<KeyPoint>& points1, vector<KeyPoint>& points2, vector<int>& indices, vector<Point3f>& points)
{
//    test();

    Mat rvec1 = Mat::zeros(3, 1, CV_32F);
    Mat tvec1 = Mat::zeros(3, 1, CV_32F);
    Mat rvec2 = Mat::zeros(3, 1, CV_32F);
    Mat tvec2 = Mat::zeros(3, 1, CV_32F);
    tvec2.at<float>(0, 0) = 1.0f;
    rvec2.at<float>(0, 0) = 0.0f;
    tvec2.at<float>(1, 0) = 0.0f;
    generateIntrinsics(intrinsics);
    Mat dist_coeffs = Mat::zeros(5, 1, CV_32F);

    vector<Point3f> planarPoints, pointCloud;
    planarPoints.resize(100);
    generatePlanarObject(planarPoints);
    pointCloud.resize(500);
    generate3DPointCloud(pointCloud);

    points = planarPoints;
    points.insert(points.end(), pointCloud.begin(), pointCloud.end());

    points1.resize(points.size());
    points2.resize(points.size());
    vector<Point2f> _points1, _points2;
    _points1.resize(points.size());
    _points2.resize(points.size());

    projectPoints(Mat(points), rvec1, tvec1, intrinsics, dist_coeffs, _points1);
    projectPoints(Mat(points), rvec2, tvec2, intrinsics, dist_coeffs, _points2);

    addPointNoise(_points1, 2.0);
    addPointNoise(_points2, 2.0);

    indices.resize(points.size());

    for(size_t i = 0; i < points1.size(); i++)
    {
        points1[i].pt = _points1[i];
        points2[i].pt = _points2[i];
        indices[i] = (int)i;

//        printf("point %d: %f %f %f\n", points[i].x, points[i].y, points[i].z);
    }

//    addLinkNoise(indices, 0.05);

    vector<Point2f> _final1, _final2;
    keyPoints2Point2f(points1, _final1);
    keyPoints2Point2f(points2, _final2);

    Rodrigues(rvec2, R);
    T = tvec2;
    Mat E = calcEssentialMatrix(intrinsics.inv(), R, tvec2);
    double error = avgSampsonusError(E, _final1, _final2);//_points1, _points2);
    printf("Initial epipolar error: %f\n", error);
}

void testReprojectPoints()
{
    Mat intrinsics, R, T;
    vector<KeyPoint> _points1, _points2;
    vector<int> indices;
    vector<Point3f> points;

    generateData(intrinsics, R, T, _points1, _points2, indices, points);

    vector<Point2f> points1, points2;
    matchesFromIndices(_points1, _points2, indices, points1, points2);

    vector<Point3f> _points;
    vector<bool> valid;
    reprojectPoints(intrinsics, R, T, points1, points2, _points, valid);
    float dist = norm(Mat(points) - Mat(_points))/sqrt(double(points.size()));

    for(size_t i = 0; i < points.size(); i++)
    {
        printf("%d %f\n", i, norm(points[i] - _points[i]));
    }
    printf("%f %f %f | %f %f %f\n", points[0].x, points[0].y, points[0].z, _points[0].x, _points[0].y, _points[0].z);
    printf("reprojectPoints test: error is %f\n", dist);
}

void generateProjections(const Mat& intrinsics, const Mat& rvec, const Mat& tvec, const vector<Point3f>& cloud, vector<KeyPoint>& keypoints)
{
  Mat distCoeffs = Mat::zeros(5, 1, CV_32F);
  vector<Point2f> imagePoints;
  imagePoints.resize(cloud.size());
  projectPoints(Mat(cloud), rvec, tvec, intrinsics, distCoeffs, imagePoints);

  for(size_t i = 0; i < imagePoints.size(); i++)
  {
    keypoints.push_back(KeyPoint(imagePoints[i], 1.0f));
  }
}

}

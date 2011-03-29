#include <posest/peh.h>
#include <sba/sba.h>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>


using namespace Eigen;
using namespace sba;
using namespace frame_common;
using namespace std;
using namespace cv;

namespace pe
{
  void invert(Mat& rvec, Mat& tvec)
  {
    Mat R, RInv;
    Rodrigues(rvec,R);
    RInv = R.inv();
    Rodrigues(RInv, rvec);
    tvec = RInv*tvec;
    tvec = tvec*(-1);
  }

  int PoseEstimatorH::estimate(const Frame& frame1, const Frame& frame2)
  {
    matches.clear();
    inliers.clear();
    cout << "Kpt size frame 1 = " << frame1.kpts.size() << endl;
    cout << "Kpt size frame 2 = " << frame2.kpts.size() << endl;
    Mat mask;
    if (windowed)
      mask = cv::windowedMatchingMask(frame1.kpts, frame2.kpts, wx, wy);
    howardMatcher->match(frame1, frame2, matches, mask);
    return estimate(frame1, frame2, matches);
  }

  Mat calcTranslation(const vector<Point3f>& points1, const vector<Point3f>& points2)
  {
    assert(points1.size() == points2.size());
    Mat t = Mat::zeros(3, 1, CV_64F);
    for (size_t i = 0; i < points1.size(); i++)
    {
      t.at<double> (0, 0) += points2[i].x - points1[i].x;
      t.at<double> (1, 0) += points2[i].y - points1[i].y;
      t.at<double> (2, 0) += points2[i].z - points1[i].z;
    }

    t /= points1.size();
    return t;
  }

  void project3dPoints(const vector<Point3f>& points, const Mat& rvec, const Mat& tvec, vector<Point3f>& modif_points)
  {
    modif_points.clear();
    modif_points.resize(points.size());
    Mat R(3, 3, CV_64FC1);
    Rodrigues(rvec, R);
    for (size_t i = 0; i < points.size(); i++)
    {
      modif_points[i].x = R.at<double> (0, 0) * points[i].x + R.at<double> (0, 1) * points[i].y + R.at<double> (0, 2)
          * points[i].z + tvec.at<double> (0, 0);
      modif_points[i].y = R.at<double> (1, 0) * points[i].x + R.at<double> (1, 1) * points[i].y + R.at<double> (1, 2)
          * points[i].z + tvec.at<double> (1, 0);
      modif_points[i].z = R.at<double> (2, 0) * points[i].x + R.at<double> (2, 1) * points[i].y + R.at<double> (2, 2)
          * points[i].z + tvec.at<double> (2, 0);
    }
  }

  int PoseEstimatorH::estimate(const Frame& f0, const Frame& f1,
                                const std::vector<cv::DMatch> &peMatches)
  {
    cout << "Matches size = " << peMatches.size() << endl;
    std::vector<cv::DMatch> matches;
    int nmatch = peMatches.size();
    for (int i=0; i<nmatch; i++)
    {
      if (f0.disps[peMatches[i].queryIdx] > minMatchDisp &&
            f1.disps[peMatches[i].trainIdx] > minMatchDisp)
        {
          matches.push_back(peMatches[i]);
        }
    }
    cout << "Matches size after disparity filtering = " << matches.size() << endl;

    vector<Point3f> opoints;
    vector<Point3f> opointsFrame2;
    vector<Point2f> ipoints;
    vector<cv::DMatch> inls;
    vector<int> indices;
    for (size_t i = 0; i < matches.size(); ++i)
    {
      if (f0.goodPts[matches[i].queryIdx] && f1.goodPts[matches[i].trainIdx])
      {
        ipoints.push_back(f1.kpts[matches[i].trainIdx].pt);
        Eigen::Vector4d vec = f0.pts[matches[i].queryIdx];
        Point3f op(vec(0), vec(1), vec(2));
        opoints.push_back(op);

        Eigen::Vector4d vec2 = f1.pts[matches[i].trainIdx];
        Point3f op2(vec2(0), vec2(1), vec2(2));
        opointsFrame2.push_back(op2);

        indices.push_back(i);
      }
    }
    bool matched = (int)matches.size() > minMatchesCount;
    if (matched)
    {
      Mat rvec, tvec;
      Mat intrinsic = Mat::zeros(Size(3, 3), CV_64F);
      intrinsic.at<double>(0, 0) = f1.cam.fx;
      intrinsic.at<double>(0, 2) = f1.cam.cx;
      intrinsic.at<double>(1, 1) = f1.cam.fy;
      intrinsic.at<double>(1, 2) = f1.cam.cy;
      intrinsic.at<double>(2, 2) = 1.0;
      solvePnP(Mat(opoints), Mat(ipoints), intrinsic, Mat::zeros(Size(1, 5), CV_64F), rvec, tvec, false);
      vector<Point2f> projectedPoints;
      projectPoints(Mat(opoints), rvec, tvec, intrinsic, Mat::zeros(Size(1, 5), CV_64F), projectedPoints);

      vector<Point3f> inliersOpoints;
      vector<Point3f> inliersOpointsFrame2;
      vector<Point2f> inliersIpoints;
      for (size_t pointInd = 0; pointInd < projectedPoints.size(); pointInd++)
      {
        double dx = ipoints[pointInd].x - projectedPoints[pointInd].x;
        double dy = ipoints[pointInd].y - projectedPoints[pointInd].y;
        double dd = f1.disps[matches[indices[pointInd]].trainIdx] - f1.cam.fx*f1.cam.tx/opoints[pointInd].z;

        if (dx*dx < maxInlierXDist2 && dy*dy < maxInlierXDist2 &&
                     dd*dd < maxInlierDDist2)
        {
          inliersIpoints.push_back(ipoints[pointInd]);
          inliersOpoints.push_back(opoints[pointInd]);
          inliersOpointsFrame2.push_back(opointsFrame2[pointInd]);
          inls.push_back(matches[indices[pointInd]]);
        }
      }
      cout << "Inliers matches size = " << inliersIpoints.size() << endl;
      if (inliersIpoints.size() > 5)
      {
        solvePnP(Mat(inliersOpoints), Mat(inliersIpoints), intrinsic, Mat::zeros(Size(1, 5), CV_64F), rvec, tvec, true);
        vector<Point3f> rotatedPoints;
        project3dPoints(Mat(inliersOpoints), rvec, tvec, rotatedPoints);
        Mat tvecAdditional = calcTranslation(rotatedPoints, inliersOpointsFrame2);
        tvec += tvecAdditional;
        cout << "tvecAdditional = " << tvecAdditional << endl;

        vector<Point2f> projectedPoints;
        projectPoints(Mat(inliersOpoints), rvec, tvec, intrinsic, Mat::zeros(Size(1, 5), CV_64F), projectedPoints);

        float reprojectionError = 0.0;
        for (size_t pointInd = 0; pointInd < projectedPoints.size(); pointInd++)
        {
          float error = norm(projectedPoints[pointInd] - inliersIpoints[pointInd]);
          if (error > reprojectionError)
          {
            reprojectionError = error;
          }
        }
        cout << "Reprojection error = " << reprojectionError << endl;
        if (reprojectionError > 2.5)
        {
          matched = false;
        }
      }
      else
      {
        inls.clear();
        matched = false;
      }

      if (matched)
      {
        if (norm(tvec) > 1.0)
        {
          inls.clear();
          return 0;
        }

        cout << rvec << endl << tvec << endl;
        invert(rvec, tvec);
        cout << rvec << endl << tvec << endl;
        Mat R;
        cv::Rodrigues(rvec, R);
        Matrix3d R2;
        Vector3d tr;
        R2 << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
        tr << tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2);
        rot = R2;
        trans = tr;
        inliers = inls;
        return inliers.size();
      }
      else
        return 0;
    }
    else
      return 0;
  }
} // ends namespace pe

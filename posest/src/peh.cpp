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
  Rodrigues(rvec, R);
  RInv = R.inv();
  Rodrigues(RInv, rvec);
  tvec = RInv * tvec;
  tvec = tvec * (-1);
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
  howardMatcher->match(frame1, frame2, matches, filteredIndices, mask);
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

int PoseEstimatorH::estimate(const Frame& f0, const Frame& f1, const std::vector<cv::DMatch> &peMatches)
{
  int nmatch = (int)peMatches.size();
  cout << "Filtered matches size = " << filteredIndices.size() << endl;
  std::vector<cv::DMatch> matches;

  for (size_t i = 0; i < filteredIndices.size(); i++)
  {
    if (f0.disps[peMatches[filteredIndices[i]].queryIdx] > minMatchDisp
        && f1.disps[peMatches[filteredIndices[i]].trainIdx] > minMatchDisp)
    {
      if (f0.goodPts[peMatches[filteredIndices[i]].queryIdx] && f1.goodPts[peMatches[filteredIndices[i]].trainIdx])
        matches.push_back(peMatches[filteredIndices[i]]);
    }
  }
  cout << "Matches size after disparity filtering = " << matches.size() << endl;
  vector<Point3f> opoints;
  vector<Point3f> opointsFrame2;
  vector<Point2f> ipoints;
  vector<cv::DMatch> inls;
  vector<int> indices;
#define CLIQUE 1
#define PNP 1

#ifdef ALL
  for (size_t i = 0; i < peMatches.size(); ++i)
  {
    if (f0.goodPts[peMatches[i].queryIdx] && f1.goodPts[peMatches[i].trainIdx])
    {
      if (f0.disps[peMatches[i].queryIdx] > minMatchDisp
              && f1.disps[peMatches[i].trainIdx] > minMatchDisp)
          {
            ipoints.push_back(f1.kpts[peMatches[i].trainIdx].pt);
            Eigen::Vector4d vec = f0.pts[peMatches[i].queryIdx];
            Point3f op(vec(0), vec(1), vec(2));
            opoints.push_back(op);

            Eigen::Vector4d vec2 = f1.pts[peMatches[i].trainIdx];
            Point3f op2(vec2(0), vec2(1), vec2(2));
            opointsFrame2.push_back(op2);

            indices.push_back(i);
          }
    }
  }
#endif
#ifdef CLIQUE
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
#endif
  bool matched = (int)matches.size() > minMatchesCount;
#ifdef PNP
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
        //inls.push_back(matches[indices[pointInd]]);
      }
    }
    cout << "Inliers matches size = " << inliersIpoints.size() << endl;
#endif

#ifdef D3D
  if (matched)
  {
    cout << "Clique size = " << opoints.size() << endl;
      vector<Vector3d> pointsFrame0(opoints.size()), pointsFrame1(opoints.size());
      Vector3d c0, c1;
      for (size_t pointInd = 0; pointInd < opoints.size(); pointInd++)
      {
        Vector3d pf0(opoints[pointInd].x, opoints[pointInd].y, opoints[pointInd].z);
        pointsFrame0[pointInd] = pf0;
        Vector3d pf1(opointsFrame2[pointInd].x, opointsFrame2[pointInd].y, opointsFrame2[pointInd].z);
        pointsFrame1[pointInd] = pf1;
        c0 += pf0;
        c1 += pf1;
      }
      c0 *= 1.0/opoints.size();
      c1 *= 1.0/opoints.size();

      Matrix3d H;
      H << 0, 0, 0,
      0, 0, 0,
      0, 0, 0;
      for (size_t pointInd = 0; pointInd < pointsFrame0.size(); pointInd++)
      {
        pointsFrame0[pointInd] -= c0;
        pointsFrame1[pointInd] -= c1;
        H += pointsFrame1[pointInd]*pointsFrame0[pointInd].transpose();
      }
      JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
      Matrix3d V = svd.matrixV();
      Matrix3d R = V * svd.matrixU().transpose();
      double det = R.determinant();
      if (det < 0.0)
      {
        //nneg++;
        V.col(2) = V.col(2)*-1.0;
        R = V * svd.matrixU().transpose();
      }
      Vector3d tr = c0-R*c1; // translation
      // transformation matrix, 3x4
      Matrix<double,3,4> tfm;
      tfm.block<3,3>(0,0) = R;
      tfm.col(3) = tr;


      // find inliers, based on image reprojection
      vector<DMatch>::iterator m;
      for (m = matches.begin(); m != matches.end(); )
      {
        Vector3d pt = tfm*f1.pts[(*m).trainIdx];
        Vector3d ipt = f0.cam2pix(pt);
        const cv::KeyPoint &kp = f0.kpts[(*m).queryIdx];
        double dx = kp.pt.x - ipt.x();
        double dy = kp.pt.y - ipt.y();
        double dd = f0.disps[(*m).queryIdx] - ipt.z();
        if (dx*dx >= 0.3 || dy*dy >= 0.3 ||
            dd*dd >= 0.5)
        {
          m = matches.erase(m);
        }
        else
        {
          cout << dx*dx << " " << dy*dy << " " << maxInlierXDist2 << " " << dd*dd << " " << maxInlierDDist2 << endl;
          m++;
        }
      }

      ipoints.clear();
      opoints.clear();
      opointsFrame2.clear();
      indices.clear();
      for (size_t i = 0; i < matches.size(); ++i)
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
      cout << "Clique size after filtering = " << opoints.size() << endl;
      if (opoints.size() < 5)
        return 0;
      pointsFrame0.resize(opoints.size());
      pointsFrame1.resize(opoints.size());
      c0 = Vector3d(0, 0, 0);
      c1 = Vector3d(0, 0, 0);
      for (size_t pointInd = 0; pointInd < opoints.size(); pointInd++)
      {
        Vector3d pf0(opoints[pointInd].x, opoints[pointInd].y, opoints[pointInd].z);
        pointsFrame0[pointInd] = pf0;
        Vector3d pf1(opointsFrame2[pointInd].x, opointsFrame2[pointInd].y, opointsFrame2[pointInd].z);
        pointsFrame1[pointInd] = pf1;
        c0 += pf0;
        c1 += pf1;
      }
      c0 *= 1.0/opoints.size();
      c1 *= 1.0/opoints.size();

      H << 0, 0, 0,
      0, 0, 0,
      0, 0, 0;
      for (size_t pointInd = 0; pointInd < pointsFrame0.size(); pointInd++)
      {
        pointsFrame0[pointInd] -= c0;
        pointsFrame1[pointInd] -= c1;
        H += pointsFrame1[pointInd]*pointsFrame0[pointInd].transpose();
      }
      JacobiSVD<Matrix3d> svdR(H, ComputeFullU | ComputeFullV);
      V = svdR.matrixV();
      R = V * svdR.matrixU().transpose();
      det = R.determinant();
      if (det < 0.0)
      {
        //nneg++;
        V.col(2) = V.col(2)*-1.0;
        R = V * svdR.matrixU().transpose();
      }
      tr = c0-R*c1; // translation
      // transformation matrix, 3x4
      tfm.block<3,3>(0,0) = R;
      tfm.col(3) = tr;


      // find inliers, based on image reprojection
      int inl = 0;
      inls.clear();
      for (int i=0; i<nmatch; i++)
      {
        Vector3d pt = tfm*f1.pts[peMatches[i].trainIdx];
        Vector3d ipt = f0.cam2pix(pt);
        const cv::KeyPoint &kp = f0.kpts[peMatches[i].queryIdx];
        double dx = kp.pt.x - ipt.x();
        double dy = kp.pt.y - ipt.y();
        double dd = f0.disps[peMatches[i].queryIdx] - ipt.z();
        if (dx*dx < maxInlierXDist2 && dy*dy < maxInlierXDist2 &&
            dd*dd < maxInlierDDist2)
        {
          if (f0.goodPts[peMatches[i].queryIdx] && f1.goodPts[peMatches[i].trainIdx])
            inls.push_back(peMatches[i]);
        }
      }
      rot = R;
      trans = tr;
      inliers = inls;
      cout << "Found " << inls.size() << " inliers" << endl;
  }
#endif

#ifdef RANSAC
  if (matched)
  {
    int bestinl = 0;
    for (int iter = 0; iter < numRansac; iter++)
    {
      vector<int> randomIndices(3);
      randomIndices[0] = rand() % opoints.size();
      randomIndices[1] = randomIndices[0];
      while (randomIndices[0] == randomIndices[1])
        randomIndices[1] = rand() % opoints.size();
      randomIndices[2] = randomIndices[0];
      while (randomIndices[0] == randomIndices[2] || randomIndices[1] == randomIndices[2])
        randomIndices[2] = rand() % opoints.size();

      vector<Vector3d> pointsFrame0(randomIndices.size()), pointsFrame1(randomIndices.size());
      Vector3d c0, c1;
      for (size_t pointInd = 0; pointInd < randomIndices.size(); pointInd++)
      {
        Vector3d pf0(opoints[randomIndices[pointInd]].x, opoints[randomIndices[pointInd]].y, opoints[randomIndices[pointInd]].z);
        pointsFrame0[pointInd] = pf0;
        Vector3d pf1(opointsFrame2[randomIndices[pointInd]].x, opointsFrame2[randomIndices[pointInd]].y,
                     opointsFrame2[randomIndices[pointInd]].z);
        pointsFrame1[pointInd] = pf1;
        c0 += pf0;
        c1 += pf1;
      }

      c0 *= 1.0 / randomIndices.size();
      c1 *= 1.0 / randomIndices.size();

      Matrix3d H;
      H << 0, 0, 0, 0, 0, 0, 0, 0, 0;
      for (size_t pointInd = 0; pointInd < pointsFrame0.size(); pointInd++)
      {
        pointsFrame0[pointInd] -= c0;
        pointsFrame1[pointInd] -= c1;
        H += pointsFrame1[pointInd] * pointsFrame0[pointInd].transpose();
      }
      JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
      Matrix3d V = svd.matrixV();
      Matrix3d R = V * svd.matrixU().transpose();
      double det = R.determinant();
      if (det < 0.0)
      {
        //nneg++;
        V.col(2) = V.col(2) * -1.0;
        R = V * svd.matrixU().transpose();
      }
      Vector3d tr = c0 - R * c1; // translation

      // transformation matrix, 3x4
      Matrix<double, 3, 4> tfm;
      tfm.block<3, 3> (0, 0) = R;
      tfm.col(3) = tr;
      // find inliers, based on image reprojection
      int inl = 0;
      for (int i = 0; i < peMatches.size(); i++)
      {
        Vector3d pt = tfm * f1.pts[peMatches[i].trainIdx];
        Vector3d ipt = f0.cam2pix(pt);
        const cv::KeyPoint &kp = f0.kpts[peMatches[i].queryIdx];
        double dx = kp.pt.x - ipt.x();
        double dy = kp.pt.y - ipt.y();
        double dd = f0.disps[peMatches[i].queryIdx] - ipt.z();
        if (dx * dx < maxInlierXDist2 && dy * dy < maxInlierXDist2 && dd * dd < maxInlierDDist2)
          inl += (int)sqrt(ipt.z()); // clever way to weight closer points
        //              inl++;
      }

#pragma omp critical
      if (inl > bestinl)
      {
        cout << "1 - 2 " << abs(norm(opoints[randomIndices[0]] - opoints[randomIndices[1]]) - norm(opointsFrame2[randomIndices[0]] - opointsFrame2[randomIndices[1]])) << endl;
        cout << "1 - 3 " << abs(norm(opoints[randomIndices[0]] - opoints[randomIndices[2]]) - norm(opointsFrame2[randomIndices[0]] - opointsFrame2[randomIndices[2]])) << endl;
        cout << "2 - 3 " << abs(norm(opoints[randomIndices[1]] - opoints[randomIndices[2]]) - norm(opointsFrame2[randomIndices[1]] - opointsFrame2[randomIndices[2]])) << endl;
        bestinl = inl;
        rot = R;
        trans = tr;
      }
    }

    inliers.clear();
    inls.clear();
    Matrix<double,3,4> tfm;
    tfm.block<3,3>(0,0) = rot;
    tfm.col(3) = trans;
    for (int i=0; i<peMatches.size(); i++)
    {
      Vector3d pt = tfm*f1.pts[peMatches[i].trainIdx];
      Vector3d ipt = f0.cam2pix(pt);
      const cv::KeyPoint &kp = f0.kpts[peMatches[i].queryIdx];
      double dx = kp.pt.x - ipt.x();
      double dy = kp.pt.y - ipt.y();
      double dd = f0.disps[peMatches[i].queryIdx] - ipt.z();
      if (dx*dx < maxInlierXDist2 && dy*dy < maxInlierXDist2 &&
          dd*dd < maxInlierDDist2)
      {
        if (f0.goodPts[peMatches[i].queryIdx] && f1.goodPts[peMatches[i].trainIdx])
          inls.push_back(peMatches[i]);
      }
    }
    inliers = inls;
    cout << "Found " << inls.size() << " inliers" << endl;
  }
#endif


#ifdef PNP
  if (inliersIpoints.size() > 5)
  {
    solvePnP(Mat(inliersOpoints), Mat(inliersIpoints), intrinsic, Mat::zeros(Size(1, 5), CV_64F), rvec, tvec, true);
    vector<Point3f> rotatedPoints;
    project3dPoints(Mat(inliersOpoints), rvec, tvec, rotatedPoints);
    //Mat tvecAdditional = calcTranslation(rotatedPoints, inliersOpointsFrame2);
    //tvec += tvecAdditional;
    //cout << "tvecAdditional = " << tvecAdditional << endl;

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
    //if (reprojectionError > 2.5)
    //{
    //  matched = false;
    //}
  }
  else
  {
    inls.clear();
    matched = false;
  }

  if (matched)
  {
    /////////////
    vector<Point3f> mopoints;
    vector<Point2f> mipoints;
    vector<int> mindices;
    for (size_t i = 0; i < peMatches.size(); ++i)
    {
      if (f0.goodPts[peMatches[i].queryIdx] && f1.goodPts[peMatches[i].trainIdx])
      {
        mipoints.push_back(f1.kpts[peMatches[i].trainIdx].pt);
        Eigen::Vector4d vec = f0.pts[peMatches[i].queryIdx];
        Point3f op(vec(0), vec(1), vec(2));
        mopoints.push_back(op);
        mindices.push_back(i);
      }
    }
    vector<Point2f> mprojectedPoints;
    projectPoints(Mat(mopoints), rvec, tvec, intrinsic, Mat::zeros(Size(1, 5), CV_64F), mprojectedPoints);
    for (size_t pointInd = 0; pointInd < mprojectedPoints.size(); pointInd++)
    {
      double dx = mipoints[pointInd].x - mprojectedPoints[pointInd].x;
      double dy = mipoints[pointInd].y - mprojectedPoints[pointInd].y;
      double dd = f1.disps[peMatches[mindices[pointInd]].trainIdx] - f1.cam.fx*f1.cam.tx/mopoints[pointInd].z;

      if (dx*dx < maxInlierXDist2 && dy*dy < maxInlierXDist2 &&
          dd*dd < maxInlierDDist2)
      {
        inls.push_back(peMatches[mindices[pointInd]]);
      }
    }
    cout << "inls size = " << inls.size() << endl;

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
    //inliers = inls;
    //return inliers.size();
  }
  else
  return 0;
}
#endif

  //else
 // return 0;

  if (polish)
  {
    cout << "Begin SBA ..." << endl;
    // system
    SysSBA sba;
    sba.verbose = 0;

    // set up nodes
    // should have a frame => node function
    Vector4d v0 = Vector4d(0, 0, 0, 1);
    Quaterniond q0 = Quaternion<double> (Vector4d(0, 0, 0, 1));
    sba.addNode(v0, q0, f0.cam, true);

    Quaterniond qr1(rot); // from rotation matrix
    Vector4d temptrans = Vector4d(trans(0), trans(1), trans(2), 1.0);

    //        sba.addNode(temptrans, qr1.normalized(), f1.cam, false);
    qr1.normalize();
    sba.addNode(temptrans, qr1, f1.cam, false);

    int in = 3;
    if (in > (int)inls.size())
      in = inls.size();

    //cout << "set up projections" << endl;
    for (int i = 0; i < (int)inls.size(); i++)
    {
      //cout << "add point" << endl;
      int i0 = inls[i].queryIdx;
      int i1 = inls[i].trainIdx;
      Vector4d pt = f0.pts[i0];
      sba.addPoint(pt);

      //cout << "projected point, ul,vl,ur" << endl;
      Vector3d ipt;
      ipt(0) = f0.kpts[i0].pt.x;
      ipt(1) = f0.kpts[i0].pt.y;
      ipt(2) = ipt(0) - f0.disps[i0];
      //cout << ipt << endl;
      sba.addStereoProj(0, i, ipt);

      //cout << "projected point frame 2" << endl;
      ipt(0) = f1.kpts[i1].pt.x;
      ipt(1) = f1.kpts[i1].pt.y;
      ipt(2) = ipt(0) - f1.disps[i1];
      sba.addStereoProj(1, i, ipt);
    }

    //cout << "before first SBA" << endl;
    sba.huber = 2.0;
    sba.doSBA(5, 10e-4, SBA_DENSE_CHOLESKY);
    //cout << "after first SBA" << endl;
    int nbad = sba.removeBad(2.0);
    //        cout << endl << "Removed " << nbad << " projections > 2 pixels error" << endl;
    //cout << "before second SBA" << endl;
    sba.doSBA(5, 10e-5, SBA_DENSE_CHOLESKY);
    //cout << "after second SBA" << endl;

    //        cout << endl << sba.nodes[1].trans.transpose().head(3) << endl;

    // get the updated transform
    trans = sba.nodes[1].trans.head(3);
    Quaterniond q1;
    q1 = sba.nodes[1].qrot;
    rot = q1.toRotationMatrix();

    // set up inliers
    inliers.clear();
    for (int i = 0; i < (int)inls.size(); i++)
    {
      ProjMap &prjs = sba.tracks[i].projections;
      if (prjs[0].isValid && prjs[1].isValid) // valid track
        inliers.push_back(inls[i]);
    }
#if 0
    printf("Inliers: %d   After polish: %d\n", (int)inls.size(), (int)inliers.size());
#endif
    return inliers.size();
  }
  return inliers.size();
}
} // ends namespace pe

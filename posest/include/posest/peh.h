#ifndef _PEH_H_
#define _PEH_H_

#include <frame_common/frame.h>
#include <boost/shared_ptr.hpp>
#include <posest/pe.h>
#include <cv.h>
#include <cstdlib>
#include <math.h>
#include <frame_common/frame.h>

namespace fc  = frame_common;
using namespace std;
namespace pe
{
class HowardDescriptorExtractor : public cv::DescriptorExtractor
{
public:
    HowardDescriptorExtractor(int _neighborhoodSize = 7): neighborhoodSize(_neighborhoodSize)
    {
      CV_Assert(neighborhoodSize/2 != 0);
    }

    virtual void read(const cv::FileNode &fn)
    {
      neighborhoodSize = fn["neighborhoodSize"];
    }
    virtual void write(cv::FileStorage &fs) const
    {
      fs << "neighborhoodSize" << neighborhoodSize;
    }

    virtual int descriptorSize() const
    {
      return neighborhoodSize*neighborhoodSize - 1;
    }
    virtual int descriptorType() const
    {
      return CV_8UC1;
    }

protected:
    int neighborhoodSize;
    virtual void computeImpl(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
    {
      cv::Mat im;
      if (image.channels() > 1)
      {
        cv::cvtColor(image, im, CV_BGR2GRAY);
      }
      else
        im = image;
      int borderSize = neighborhoodSize/2;
      //temporary removed. I need that keypoints count doesn't change.
      //removeBorderKeypoints(keypoints, im.size(), borderSize);
      descriptors.create(keypoints.size(), descriptorSize(), descriptorType());
      descriptors.setTo(cv::Scalar(0));

      for (size_t keypointInd = 0; keypointInd < keypoints.size(); keypointInd++)
      {
        int index = 0;
        for(int y = keypoints[keypointInd].pt.y - borderSize; y < keypoints[keypointInd].pt.y + borderSize; y++)
        {
          for(int x = keypoints[keypointInd].pt.x - borderSize; x < keypoints[keypointInd].pt.x + borderSize; x++)
          {
            if (x == keypoints[keypointInd].pt.x && y == keypoints[keypointInd].pt.y)
            {
              continue;
            }
            //TODO: fix it
            if (x < 0 || x >= image.cols || y < 0 || y >= image.rows)
              continue;
            descriptors.at<unsigned char>(keypointInd, index++) = im.at<unsigned char>(y, x);
          }
        }
      }
    }
};


class HowardStereoMatcher
{
public:
  HowardStereoMatcher(float thresh, int descriptorSize): threshold(thresh), descriptorSize(descriptorSize)
  {
    extractor = new HowardDescriptorExtractor(descriptorSize);
  }

  void match(const frame_common::Frame& prevFrame, const frame_common::Frame& frame,
             vector<cv::DMatch>& matches, vector<int>& filteredIndices, const cv::Mat& mask)
  {
    if (mask.empty())
    {
      windowedMask = cv::Mat(prevFrame.kpts.size(), frame.kpts.size(), CV_8UC1, cv::Scalar::all(1));
    }
    else
    {
      mask.copyTo(windowedMask);
    }

    extractor->compute(prevFrame.img, const_cast<vector<cv::KeyPoint>&>(prevFrame.kpts), prevFrameDtors);
    filterKpts(prevFrame.img, prevFrame.kpts, true);
    extractor->compute(frame.img, const_cast<vector<cv::KeyPoint>&>(frame.kpts), frameDtors);
    filterKpts(frame.img, frame.kpts, false);

    cv::Mat scoreMatrix;
    calculateScoreMatrix(scoreMatrix);
    calculateCrossCheckMatches(scoreMatrix, matches);

    cout << "After crosscheck = " << matches.size() << endl;
    if (matches.size())
    {
      cv::Mat consistMatrix;
      calculateConsistMatrix(matches, prevFrame, frame, consistMatrix);
      filterMatches(consistMatrix, filteredIndices);
    }
    cout << "After filtering = " << filteredIndices.size() << endl;
  }

private:
  void filterKpts(const cv::Mat& img, const vector<cv::KeyPoint>& kpts, bool orientation)
  {
    int border = descriptorSize/2;
    for (size_t ind = 0; ind < kpts.size(); ind++)
    {
      const cv::Point& kpt = kpts[ind].pt;
      if (kpt.x < border || kpt.x >= img.cols - border || kpt.y < border || kpt.y >= img.rows - border)
      {
        if (orientation)
          windowedMask.row(ind).setTo(cv::Scalar::all(0));
        else
          windowedMask.col(ind).setTo(cv::Scalar::all(0));
      }
    }
  }

  void calculateScoreMatrix(cv::Mat& scoreMatrix)
  {
    scoreMatrix.create(prevFrameDtors.rows, frameDtors.rows, CV_32S);
    scoreMatrix.setTo(-1);

    for (int row = 0; row < prevFrameDtors.rows; row++)
      for (int col = 0; col < frameDtors.rows; col++)
      {
        if (windowedMask.at<unsigned char>(row, col))
        {
          //calculate SAD between row descriptor from first image and col descriptor from second image
          scoreMatrix.at<int>(row, col) = cv::sum(cv::abs((prevFrameDtors.row(row) - frameDtors.row(col))))[0];
        }
      }
  }

  void calculateCrossCheckMatches(const cv::Mat& scoreMatrix, vector<cv::DMatch>& matches)
  {
    cv::Point minIndex;
    vector<int> matches1to2(scoreMatrix.rows);
    for (int row = 0; row < scoreMatrix.rows; row++)
    {
      cv::minMaxLoc(scoreMatrix.row(row), 0, 0, &minIndex, 0, windowedMask.row(row));
      matches1to2[row] = minIndex.x;
    }
    vector<int> matches2to1(scoreMatrix.cols);
    for (int col = 0; col < scoreMatrix.cols; col++)
    {
      cv::minMaxLoc(scoreMatrix.col(col), 0, 0, &minIndex, 0, windowedMask.col(col));
      matches2to1[col] = minIndex.y;
    }
#if 0
    for (size_t mIndex = 0; mIndex < matches1to2.size(); mIndex++)
    {
      if (matches2to1[matches1to2[mIndex]] == (int)mIndex)
      {
        matches.push_back(cv::DMatch(mIndex, matches1to2[mIndex], 0.f));
      }
    }
#endif

    for (size_t mIndex = 0; mIndex < matches1to2.size(); mIndex++)
    {
      matches.push_back(cv::DMatch(mIndex, matches1to2[mIndex], 0.f));
    }

    for (size_t mIndex = 0; mIndex < matches2to1.size(); mIndex++)
    {
      if (matches1to2[matches2to1[mIndex]] != mIndex)
        matches.push_back(cv::DMatch(matches2to1[mIndex], mIndex, 0.f));
    }
  }

  double calcDeltaL(const cv::Point3f& p11, const cv::Point3f& p21, double t, double f, double threshold)
  {
    double A = pow( (p11.x - p21.x)*(t - p11.x) - (p11.y - p21.y)*p11.y - (p11.z - p21.z)*p11.z, 2);
    double B = pow( (p11.x - p21.x)*p11.x - (p11.y - p21.y)*p11.y - (p11.z - p21.z)*p11.z, 2);
    double C = 0.5*pow(t* (p11.y - p21.y), 2);
    double D = pow( (p11.x - p21.x)*(t - p21.x) - (p11.y - p21.y)*p21.y - (p11.z - p21.z)*p21.z, 2);
    double E = pow( (p11.x - p21.x)*p21.x - (p11.y - p21.y)*p21.y - (p11.z - p21.z)*p21.z, 2);
    double F = C;
    cv::Point3f diff = p11 - p21;
    double L = cv::norm(diff);
    return threshold*sqrt(p11.z*p11.z*(A+B+C) + p21.z*p21.z*(D+E+F)) / (L*f);
  }

  void calculateConsistMatrix(const vector<cv::DMatch>& matches, const frame_common::Frame& prevFrame,
                              const frame_common::Frame& frame, cv::Mat& consistMatrix)
  {
    consistMatrix.create(matches.size(), matches.size(), CV_8UC1);
    for (int row = 0; row < consistMatrix.rows; row++)
    {
      if (!prevFrame.goodPts[matches[row].queryIdx])
      {
        consistMatrix.row(row).setTo(cv::Scalar(0));
        consistMatrix.col(row).setTo(cv::Scalar(0));
      }
      else
      {
        for (int col = 0; col < row; col++)
        {
          unsigned char consistent = 0;
          if (frame.goodPts[matches[col].trainIdx])
          {
            Eigen::Vector4d vec = prevFrame.pts[matches[row].queryIdx];
            cv::Point3f p11(vec(0), vec(1), vec(2));
            vec = prevFrame.pts[matches[col].queryIdx];
            cv::Point3f p21(vec(0), vec(1), vec(2));

            vec = frame.pts[matches[row].trainIdx];
            cv::Point3f p12(vec(0), vec(1), vec(2));
            vec = frame.pts[matches[col].trainIdx];
            cv::Point3f p22(vec(0), vec(1), vec(2));

            cv::Point3f diff1 = p11 - p21;
            cv::Point3f diff2 = p12 - p22;
            //if (abs(norm(diff1) - norm(diff2)) < threshold)
            //  consistent = 1;
            double L1 = cv::norm(diff1), L2 = cv::norm(diff2);
            double dL1 = calcDeltaL(p11, p21, prevFrame.cam.tx, prevFrame.cam.fx, threshold);
            double dL2 = calcDeltaL(p12, p22, frame.cam.tx, frame.cam.fx, threshold);
            if (fabs(L1 - L2) < 3*sqrt(dL1*dL1+dL2*dL2))
              consistent = 1;

          }
          consistMatrix.at<unsigned char>(row, col) = consistMatrix.at<unsigned char>(col, row) = consistent;
        }
      }
      consistMatrix.at<unsigned char>(row, row) = 1;
    }
  }

  void filterMatches(const cv::Mat& consistMatrix, vector<int>& filteredIndices)
  {
    std::vector<int> indices;
    //initialize clique
    cv::Mat sums(1, consistMatrix.rows, CV_32S);
    for (int row = 0; row < consistMatrix.rows; row++)
      sums.at<int>(0, row) = cv::sum(consistMatrix.row(row))[0];
    cv::Point maxIndex;
    cv::minMaxLoc(sums, 0, 0, 0, &maxIndex, cv::Mat());
    indices.push_back(maxIndex.x);

    int lastAddedIndex = maxIndex.x;

    //initialize compatible matches
    vector<int> compatibleMatches, sizes;
    for (int mIndex = 0; mIndex < consistMatrix.cols; mIndex++)
    {
      if (consistMatrix.at<unsigned char>(lastAddedIndex, mIndex))
      {
        compatibleMatches.push_back(mIndex);
        sizes.push_back(sum(consistMatrix.row(mIndex))[0]);
      }
    }

    while(true)
    {
      vector<int>::iterator cmIter, sizeIter;
      if (lastAddedIndex != maxIndex.x)
      {
        for (cmIter = compatibleMatches.begin(), sizeIter = sizes.begin();
             cmIter != compatibleMatches.end(), sizeIter != sizes.end();)
        {
          if (consistMatrix.at<unsigned char>(lastAddedIndex, *cmIter))
          {
            cmIter++;
            sizeIter++;
          }
          else
          {
            cmIter = compatibleMatches.erase(cmIter);
            sizeIter = sizes.erase(sizeIter);
          }
        }
      }
      if (!compatibleMatches.size())
        break;
      vector<int>::iterator maxIter = compatibleMatches.end(), maxSizeIter = sizes.end();
      int maxSize = 0;
      for (cmIter = compatibleMatches.begin(), sizeIter = sizes.begin();
                 cmIter != compatibleMatches.end(), sizeIter != sizes.end(); cmIter++, sizeIter++)
      {
        if (*sizeIter > maxSize)
        {
          maxSize = *sizeIter;
          maxSizeIter = sizeIter;
          maxIter = cmIter;
        }
      }
      indices.push_back(*maxIter);
      lastAddedIndex = *maxIter;
      compatibleMatches.erase(maxIter);
      sizes.erase(maxSizeIter);
    }

    filteredIndices = indices;
  }

private:
  float threshold;
  int descriptorSize;
  cv::Mat prevFrameDtors, frameDtors;
  cv::Ptr<cv::DescriptorExtractor> extractor;
  cv::Mat windowedMask;
};

class PoseEstimatorH : public PoseEstimator
{
public:
  using PoseEstimator::estimate;

  PoseEstimatorH(int NRansac, bool LMpolish, double mind,
                  double maxidx, double maxidd, float matcherThreshold, int minMatchesCount, int descriptorSize) :
                    PoseEstimator(NRansac, LMpolish, mind, maxidx, maxidd), minMatchesCount(minMatchesCount)
  {
    usedMethod = Stereo;
    howardMatcher = new HowardStereoMatcher(matcherThreshold, descriptorSize);
  };
  ~PoseEstimatorH() { };

  virtual int estimate(const fc::Frame& frame1, const fc::Frame& frame2);

  virtual int estimate(const fc::Frame& frame1, const fc::Frame& frame2,
                       const std::vector<cv::DMatch> &matches);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // needed for 16B alignment
private:
  cv::Ptr<HowardStereoMatcher> howardMatcher;
  int minMatchesCount;
  std::vector<int> filteredIndices;
};

} // ends namespace pe
#endif // _PEH_H_

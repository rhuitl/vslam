/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/


#ifndef _PE_H_
#define _PE_H_

#include <frame_common/frame.h>
#include <boost/shared_ptr.hpp>
#include <cv.h>
#include <cstdlib>
#include <math.h>

namespace fc  = frame_common;

namespace pe
{
  /// @todo This should go in OpenCV
  void drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& keypoints1,
                   const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints2,
                   const std::vector<Match>& matches, cv::Mat& outImg,
                   const cv::Scalar& matchColor = cv::Scalar::all(-1),
                   const cv::Scalar& singlePointColor = cv::Scalar::all(-1),
                   const std::vector<char>& matchesMask = std::vector<char>(),
                   int flags = cv::DrawMatchesFlags::DEFAULT);

  /// A class that estimates camera pose from features in image frames.
  class PoseEstimator
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // needed for 16B alignment

    enum MethodType
    {
      SFM = 0,
      PnP = 1,
      Stereo = 2
    };

    MethodType usedMethod;

    // true if we want to SVD polish the result
    bool polish;

    // all matches and inliers
    std::vector<Match> matches; ///< Matches between features in frames.
    std::vector<Match> inliers; ///< RANSAC inliers of matches.

    /// number of RANSAC iterations
    int numRansac;
    
    /// Whether to do windowed or whole-image matching.
    int windowed;

    /// Descriptor matcher to use (features_2d::BruteForceMatcher by default).
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int wx /*< Width of matching window.*/, wy /*< Height of matching window*/;

    PoseEstimator(int NRansac, bool LMpolish, double mind,
                  double maxidx, double maxidd);
    ~PoseEstimator() { }

    /// Minimum disparity for RANSAC point matches.
    double minMatchDisp;

    /// Maximum dist^2 for inliers in pixels.
    double maxInlierXDist2, maxInlierDDist2;

    /// Set the Descriptor Matcher to use to match features between frames.
    void setMatcher(const cv::Ptr<cv::DescriptorMatcher>& matcher);

    /// \brief Uses RANSAC to find best inlier count from provided matches, 
    /// optionally polishes the result.
    /// Frames must have filled features and descriptors.
    virtual int estimate(const fc::Frame& frame1, const fc::Frame& frame2,
                         const std::vector<Match> &matches) = 0;

    /// \brief Uses RANSAC to find best inlier count by finding matches using 
    /// the provided matcher, optionally polishes the result.
    /// Frames must have filled features and descriptors.
    virtual int estimate(const fc::Frame& frame1, const fc::Frame& frame2);

    // Helper function to perform polishing. (This was not being used anywhere?)
    //void doPolish(fc::Frame& f0, fc::Frame& f1);

    void setTestMode(bool mode);
    void setTestMatches(const std::vector<Match>& matches)
    {
      assert(testMode==true);
      testMatches = matches;
    };

    /// Get the method used for inlier detection.
    MethodType getMethod() const {return usedMethod;};

    // transform
    Eigen::Matrix3d rot; ///< Rotation matrix of camera between the frames.
    Eigen::Vector3d trans; ///< Translation of the camera between the frames.

  protected:
    void matchFrames(const fc::Frame& f0, const fc::Frame& f1, std::vector<int>& fwd_matches);
    
    bool testMode;
    std::vector<Match> testMatches;

  };



} // ends namespace pe
#endif // _PE3D_H_

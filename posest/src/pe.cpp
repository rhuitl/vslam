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


#include <posest/pe.h>
#include <sba/sba.h>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <iostream>


using namespace Eigen;
using namespace sba;
using namespace frame_common;
using namespace std;

namespace pe
{

  void drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& keypoints1,
                   const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints2,
                   const std::vector<Match>& matches, cv::Mat& outImg,
                   const cv::Scalar& matchColor, const cv::Scalar& singlePointColor,
                   const std::vector<char>& matchesMask, int flags)
  {
    using namespace cv;
    Size size( img1.cols + img2.cols, MAX(img1.rows, img2.rows) );
    if( flags & DrawMatchesFlags::DRAW_OVER_OUTIMG )
    {
        if( size.width > outImg.cols || size.height > outImg.rows )
            CV_Error( CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
    }
    else
    {
        outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
        Mat outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        cvtColor( img1, outImg1, CV_GRAY2RGB );
        Mat outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
        cvtColor( img2, outImg2, CV_GRAY2RGB );
    }

    RNG rng;
    // draw keypoints
    if( !(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) )
    {
        bool isRandSinglePointColor = singlePointColor == Scalar::all(-1);
        for( vector<KeyPoint>::const_iterator it = keypoints1.begin(); it < keypoints1.end(); ++it )
        {
            circle( outImg, it->pt, 3, isRandSinglePointColor ?
                    Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)) : singlePointColor );
        }
        for( vector<KeyPoint>::const_iterator it = keypoints2.begin(); it < keypoints2.end(); ++it )
        {
            cv::Point p = it->pt;
            circle( outImg, cv::Point(p.x+img1.cols, p.y), 3, isRandSinglePointColor ?
                    Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)) : singlePointColor );
        }
     }

    // draw matches
    bool isRandMatchColor = matchColor == Scalar::all(-1);
    if( !matchesMask.empty() && matchesMask.size() != matches.size() )
        CV_Error( CV_StsBadSize, "mask must have the same size as matches" );
    for( int i = 0; i < matches.size(); i++ )
    {
        if( matchesMask.empty() || matchesMask[i] )
        {
            Point2f pt1 = keypoints1[matches[i].index1].pt,
                    pt2 = keypoints2[matches[i].index2].pt,
                    dpt2 = Point2f( std::min(pt2.x+img1.cols, float(outImg.cols-1)), pt2.y );
            Scalar randColor( rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256) );
            circle( outImg, pt1, 3, isRandMatchColor ? randColor : matchColor );
            circle( outImg, dpt2, 3, isRandMatchColor ? randColor : matchColor );
            line( outImg, pt1, dpt2, isRandMatchColor ? randColor : matchColor );
        }
    }
  }

  PoseEstimator::PoseEstimator(int NRansac, bool LMpolish, double mind,
                                   double maxidx, double maxidd) : testMode(false)
  {
    polish = LMpolish;
    numRansac = NRansac;
    minMatchDisp = mind;
    maxInlierXDist2 = maxidx*maxidx;
    maxInlierDDist2 = maxidd*maxidd;
    rot.setIdentity();
    trans.setZero();
    
    // matcher
    matcher = new cv::BruteForceMatcher< cv::L2<float> >;
    wx = 92; wy = 48;
    windowed = true;
  }

void PoseEstimator::matchFrames(const fc::Frame& f0, const fc::Frame& f1, std::vector<int>& fwd_matches)
  {
    cv::Mat mask;
    if (windowed)
      mask = cv::windowedMatchingMask(f0.kpts, f1.kpts, wx, wy);

    matcher->clear();
    matcher->add(f1.dtors);
    matcher->match(f0.dtors, mask, fwd_matches);
  }

  //
  // find the best estimate for a geometrically-consistent match
  //   sets up frames internally using sparse stereo
  // NOTE: we do forward/reverse matching and save all unique matches
  //   sometimes forward works best, sometimes reverse
  // uses the SVD procedure for aligning point clouds
  // final SVD polishing step
  //

  int PoseEstimator::estimate(const Frame &f0, const Frame &f1)
  {
    // set up match lists
    matches.clear();
    inliers.clear();

    // do forward and reverse matches
    std::vector<int> fwd_matches, rev_matches;
    matchFrames(f0, f1, fwd_matches);
    matchFrames(f1, f0, rev_matches);
    //printf("**** Forward matches: %d, reverse matches: %d ****\n", (int)fwd_matches.size(), (int)rev_matches.size());

    // combine unique matches into one list
    for (int i = 0; i < (int)fwd_matches.size(); ++i) {
      if (fwd_matches[i] >= 0)
        matches.push_back( Match(i, fwd_matches[i]) );
    }
    for (int i = 0; i < (int)rev_matches.size(); ++i) {
      if (rev_matches[i] >= 0 && i != fwd_matches[rev_matches[i]])
        matches.push_back( Match(rev_matches[i], i) );
    }
    //printf("**** Total unique matches: %d ****\n", (int)matches.size());
    
    // do it
    return estimate(f0, f1, matches);
  }

  void PoseEstimator::setMatcher(const cv::Ptr<cv::DescriptorMatcher>& new_matcher)
  {
    matcher = new_matcher;
  }
  
  void PoseEstimator::setTestMode(bool mode)
  {
    testMode = mode;
  };
}

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


#include <frame_common/frame.h>

using namespace Eigen;
using namespace std;

// elapsed time in milliseconds
#include <sys/time.h>
static double mstime()
{
  timeval tv;
  gettimeofday(&tv,NULL);
  long long ts = tv.tv_sec;
  ts *= 1000000;
  ts += tv.tv_usec;
  return (double)ts*.001;
}

namespace frame_common
{

  void Frame::setCamParams(const CamParams &c)
  {
    // monocular
    cam = c;
    iproj.setZero();
    iproj(0,0) = c.fx;
    iproj(1,1) = c.fy;
    iproj(0,2)=c.cx;
    iproj(1,2)=c.cy;
    iproj(2,2)=1.0;

    // stereo
    disp_to_cart(0,0) = 1/c.fx;
    disp_to_cart(1,1) = 1/c.fy;
    disp_to_cart(0,3) = -c.cx/c.fx;
    disp_to_cart(1,3) = -c.cy/c.fy;
    disp_to_cart(2,3) = 1.0;
    disp_to_cart(3,2) = 1/(c.tx*c.fx);

    cart_to_disp.setZero();
    cart_to_disp(0,0) = c.fx;
    cart_to_disp(1,1) = c.fy;
    cart_to_disp(0,2) = c.cx;
    cart_to_disp(1,2) = c.cy;
    cart_to_disp(2,3) = c.fx*c.tx;
    cart_to_disp(3,2) = 1.0;

    if (c.tx > 0.0)
      isStereo = true;
    else
      isStereo = false;
  }


  // return x,y,d from X,Y,Z
  Eigen::Vector3d Frame::cam2pix(const Eigen::Vector3d &cam_coord) const
  {
    double xl = cam.fx * cam_coord[0] + cam.cx * cam_coord[2];
    double y = cam.fy * cam_coord[1] + cam.cy * cam_coord[2];
    double w = cam_coord[2];
    double xd = cam.fx * cam.tx;
    double rw = 1.0/w;
    double y_rw = y * rw;
    return Eigen::Vector3d(xl*rw, y_rw, xd*rw);
  }

  // return X,Y,Z from x,y,d
  Eigen::Vector3d Frame::pix2cam(const Eigen::Vector3d &pix_coord) const
  {
    double x = pix_coord[0] - cam.cx;
    double y = pix_coord[1] - cam.cy;
    double z = cam.fx;
    double w = pix_coord[2]/cam.tx;
    return Eigen::Vector3d(x/w, y/w, z/w);
  }


  // frame processor

  FrameProc::FrameProc(int v)
  {
    // detector (by default, FAST)
    detector = new cv::GridAdaptedFeatureDetector(new cv::FastFeatureDetector(v), 1000);
    // detector = new cv::FastFeatureDetector(v);

    // descriptor (by default, SURF)
    extractor = new cv::SurfDescriptorExtractor;

    // stereo
    ndisp = 64;
    doSparse = false;           // use dense stereo by default
  }

  void FrameProc::setFrameDetector(const cv::Ptr<cv::FeatureDetector>& new_detector)
  {
    detector = new_detector;
  }

  void FrameProc::setFrameDescriptor(const cv::Ptr<cv::DescriptorExtractor>& new_extractor)
  {
    extractor = new_extractor;
  }

  // set up mono frame
  void FrameProc::setMonoFrame(Frame& frame, const cv::Mat &img)
  {
    frame.img = img;

    // set keypoints and descriptors
    frame.kpts.clear();
    //double t0 = mstime();
    detector->detect(img, frame.kpts);
    //double t1 = mstime();
    extractor->compute(img, frame.kpts, frame.dtors);
    //double t2 = mstime();

    frame.pts.resize(frame.kpts.size());
    frame.goodPts.assign(frame.kpts.size(), false);
    frame.disps.assign(frame.kpts.size(), 10);
  }

  // set up stereo frame
  // assumes frame has camera params already set
  // <nfrac> is nonzero if <imgr> is a dense stereo image
  void FrameProc::setStereoFrame(Frame &frame, const cv::Mat &img, const cv::Mat &imgr, int nfrac)
  {
    setMonoFrame(frame, img);

    frame.imgRight = imgr;

    // set stereo
    setStereoPoints(frame,nfrac);
    //double t3 = mstime();

    //  printf("detect %0.2f  extract %0.2f  stereo %0.2f \n", t1-t0, t2-t1, t3-t2);

  }


  // set up stereo points
  void FrameProc::setStereoPoints(Frame &frame, int nfrac)
  {
    //  if (img.rows == 0 || imgRight.rows == 0)
    //    return;
    frame.disps.clear();


    FrameStereo *st;
    if (doSparse)
      st = new SparseStereo(frame.img,frame.imgRight,true,ndisp);
    else if (nfrac > 0)
      { 
        DenseStereo::ndisp = ndisp;
        st = new DenseStereo(frame.img,frame.imgRight,ndisp,1.0/(double)nfrac);
      }
    else
      { 
        DenseStereo::ndisp = ndisp;
        st = new DenseStereo(frame.img,frame.imgRight,ndisp);
      }

    int nkpts = frame.kpts.size();
    frame.goodPts.resize(nkpts);
    frame.pts.resize(nkpts);
    frame.disps.resize(nkpts);
    for (int i=0; i<nkpts; i++)
      {
	      double disp = st->lookup_disparity(frame.kpts[i].pt.x,frame.kpts[i].pt.y);
	      frame.disps[i] = disp;
	      if (disp > 0.0)           // good disparity
	        {
	          frame.goodPts[i] = true;
	          Vector3d pt(frame.kpts[i].pt.x,frame.kpts[i].pt.y,disp);
	          frame.pts[i].start(3) = frame.pix2cam(pt);
	          frame.pts[i](3) = 1.0;
	          //          cout << pts[i].transpose() << endl;
	        }
	      else
	        frame.goodPts[i] = false;
      }
    delete st;
  }

} // end namespace frame_common

  // Detect keypoints
  // this returns no keypoints on my sample images...
  // boost::shared_ptr<f2d::FeatureDetector> detector( new f2d::StarFeatureDetector(16, 100) );
  //  boost::shared_ptr<f2d::FeatureDetector> detector( new f2d::StarFeatureDetector );
  //  boost::shared_ptr<f2d::FeatureDetector> detector( new f2d::SurfFeatureDetector(4000.0) );
  //  boost::shared_ptr<f2d::FeatureDetector> detector( new f2d::HarrisFeatureDetector(300, 5) );
  //  boost::shared_ptr<f2d::FeatureDetector> detector( new f2d::FastFeatureDetector(50) );

  // Compute descriptors
  //  boost::shared_ptr<f2d::DescriptorExtractor> extractor(cd); // this is the calonder descriptor
  //  boost::shared_ptr<f2d::DescriptorExtractor> extractor( new f2d::SurfDescriptorExtractor(3, 4, true) );

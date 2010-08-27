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
using namespace pcl;

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
  
    void PointcloudProc::setPointcloud(Frame &frame, const pcl::PointCloud<pcl::PointXYZRGB>& input_cloud)
    {
      reduceCloud(input_cloud, frame.pointcloud);
      
      // For now, let's keep a 1-1 mapping between pl_pts, keypts, etc., etc.
      // Basically replicating all the info in the pointcloud but whatever.
      // TODO: Do something more intelligent than this.
      frame.pl_pts.clear();
      frame.pl_kpts.clear();
      frame.pl_normals.clear();
      frame.pl_ipts.clear();
      
      unsigned int ptcloudsize = frame.pointcloud.points.size();
      frame.pl_pts.resize(ptcloudsize);
      frame.pl_kpts.resize(ptcloudsize);
      frame.pl_normals.resize(ptcloudsize);
      frame.pl_ipts.resize(ptcloudsize);
      
      for (unsigned int i=0; i < frame.pointcloud.points.size(); i++)
      {
        PointXYZRGBNormal &pt = frame.pointcloud.points[i];
        
        frame.pl_pts[i] = Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
        frame.pl_normals[i] = Eigen::Vector4d(pt.normal[0], pt.normal[1], pt.normal[2], 1.0);
        frame.pl_kpts[i] = projectPoint(frame.pl_pts[i], frame.cam);
        frame.pl_ipts[i] = -1;
      }
    }
    
    void PointcloudProc::match(const Frame& frame0, const Frame& frame1, 
          const Eigen::Vector3d& trans, const Eigen::Quaterniond& rot, 
          std::vector<pe::Match>& matches)
    {
      PointCloud<PointXYZRGBNormal> transformed_cloud;
      
      // First, transform the current frame. (Is this inverse?) (Or just transform the other cloud?)
      //transformPointCloudWithNormals<PointXYZRGBNormal>(frame1.cloud, transformed_cloud, -trans.cast<float>(), rot.cast<float>().conjugate());
      
      //transformPointCloudWithNormals<PointXYZRGBNormal>(frame0.pointcloud, transformed_cloud, -trans.cast<float>(), rot.cast<float>().conjugate());
      transformPointCloudWithNormals<PointXYZRGBNormal>(frame0.pointcloud, transformed_cloud, Vector3f(0,0,0), rot.cast<float>().conjugate());
      transformPointCloudWithNormals<PointXYZRGBNormal>(transformed_cloud, transformed_cloud, -trans.cast<float>(), Quaternionf(1, 0, 0, 0));
      //pcl::io::savePCDFileASCII ("cloud0.pcd", transformed_cloud);
      //pcl::io::savePCDFileASCII ("cloud1.pcd", frame1.pointcloud);
      
      // Optional/TODO: Perform ICP to further refine estimate.
      /*PointCloud<PointXYZRGBNormal> cloud_reg;

      IterativeClosestPointNonLinear<PointXYZRGBNormal, PointXYZRGBNormal> reg;
      reg.setInputCloud (boost::make_shared<const PointCloud<PointXYZRGBNormal> > (transformed_cloud));
      reg.setInputTarget (boost::make_shared<const PointCloud<PointXYZRGBNormal> > (frame1.pointcloud));
      reg.setMaximumIterations(50);
      reg.setTransformationEpsilon (1e-8);

      reg.align(cloud_reg); */
            
      // Find matches between pointclouds in frames. (TODO: also compare normals)
      std::vector<int> f0_indices, f1_indices;
      getMatchingIndices(transformed_cloud, frame1.pointcloud, f0_indices, f1_indices);
      
      // Fill in keypoints and projections of relevant features.
      // Currently just done when setting the pointcloud.
      
      // Convert matches into the correct format.
      matches.clear();
      // Starting at i=1 as a hack to not let through (0,0,0) matches (why is this in the ptcloud?))
      for (unsigned int i=1; i < f0_indices.size(); i++)
      {           
        const PointXYZRGBNormal &pt0 = transformed_cloud.points[f0_indices[i]];
        const PointXYZRGBNormal &pt1 = frame1.pointcloud.points[f1_indices[i]];
        
        // Figure out distance and angle between normals
        Quaterniond normalquat;
        Vector3d norm0(pt0.normal[0], pt0.normal[1], pt0.normal[2]), norm1(pt1.normal[0], pt1.normal[1], pt1.normal[2]);
        normalquat.setFromTwoVectors(norm0, norm1);
        //double angledist = normalquat.angularDistance(normalquat.Identity());
        double dist = (Vector3d(pt0.x, pt0.y, pt0.z)-Vector3d(pt1.x, pt1.y, pt1.z)).norm();
        
        /* Vector4d p0_pt = Vector4d(pt0.x, pt0.y, pt0.z, 1.0);
        Vector3d expected_proj = projectPoint(p0_pt, frame0.cam);
        
        Vector3d diff = expected_proj - frame1.pl_kpts[f1_indices[i]].start<3>();
        diff(2) = diff(2) - diff(0);
        
        printf("[Proj difference] %f %f %f\n", diff(0), diff(1), diff(2)); */
        
        if ((norm0 - norm1).norm() < 0.5 && dist < 0.2)
          matches.push_back(pe::Match(f0_indices[i], f1_indices[i], dist));
      }
      
      printf("[FrameExtended] Found %d matches, then converted %d matches.\n", (int)f0_indices.size(), (int)matches.size());
    }
    
    void PointcloudProc::getMatchingIndices(const PointCloud<PointXYZRGBNormal>& input, 
              const PointCloud<PointXYZRGBNormal>& output, 
              std::vector<int>& input_indices, std::vector<int>& output_indices)
    {
      // TODO: Don't calculate the KDTree each time.
      KdTreeANN<PointXYZRGBNormal> input_tree, output_tree;
        
      input_tree.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGBNormal> >(input));
      output_tree.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGBNormal> >(output));
      
      // Iterate over the output tree looking for all the input points and finding
      // nearest neighbors.
      for (unsigned int i = 0; i < input.points.size(); i++)
      {
        PointXYZRGBNormal input_pt = input.points[i];
        std::vector<int> input_indexvect(1), output_indexvect(1); // Create a vector of size 1.
        std::vector<float> input_distvect(1), output_distvect(1);
        
        // Find the nearest neighbor of the input point in the output tree.
        output_tree.nearestKSearch(input_pt, 1, input_indexvect, input_distvect);
        
        PointXYZRGBNormal output_pt = output.points[input_indexvect[0]];
        
        // Find the nearest neighbor of the output point in the input tree.
        input_tree.nearestKSearch(output_pt, 1, output_indexvect, output_distvect);
        
        // If they match, add them to the match vectors.
        if (output_indexvect[0] == (int)i)
        {
          input_indices.push_back(i);
          output_indices.push_back(input_indexvect[0]);
        }
      }
    }
    
    // Subsample cloud for faster matching and processing, while filling in normals.
    void PointcloudProc::reduceCloud(const PointCloud<PointXYZRGB>& input, PointCloud<PointXYZRGBNormal>& output)
    {
      PointCloud<PointXYZRGB> cloud_nan_filtered, cloud_box_filtered, cloud_voxel_reduced;
      PointCloud<Normal> normals;
      PointCloud<PointXYZRGBNormal> cloud_normals;
      
      std::vector<int> indices;
      
      // Filter out nans.
      removeNaNFromPointCloud(input, cloud_nan_filtered, indices);
      indices.clear();
      
      // Filter out everything outside a [200x200x200] box.
      Eigen::Vector4f min_pt(-100, -100, -100, -100);
      Eigen::Vector4f max_pt(100, 100, 100, 100);
      getPointsInBox(cloud_nan_filtered, min_pt, max_pt, indices);
      
      ExtractIndices<PointXYZRGB> boxfilter;
      boxfilter.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGB> >(cloud_nan_filtered));
      boxfilter.setIndices (boost::make_shared<vector<int> > (indices));
      boxfilter.filter(cloud_box_filtered);
      
      // Reduce pointcloud
      VoxelGrid<PointXYZRGB> voxelfilter;
      voxelfilter.setInputCloud (boost::make_shared<const PointCloud<PointXYZRGB> > (cloud_box_filtered));
      voxelfilter.setLeafSize (0.05, 0.05, 0.05);
      voxelfilter.filter (cloud_voxel_reduced);
      
      // Compute normals
      NormalEstimation<PointXYZRGB, Normal> normalest;
      normalest.setViewPoint(0, 0, 0);
      normalest.setSearchMethod (boost::make_shared<KdTreeANN<PointXYZRGB> > ());
      //normalest.setKSearch (10);
      normalest.setRadiusSearch (0.25);
      normalest.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGB> >(cloud_voxel_reduced));
      normalest.compute(normals);
      
      pcl::concatenateFields (cloud_voxel_reduced, normals, cloud_normals);

      // Filter based on curvature
      PassThrough<PointXYZRGBNormal> normalfilter;
      normalfilter.setFilterFieldName("curvature");
      normalfilter.setFilterLimits(0.0, 0.2);
      normalfilter.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGBNormal> >(cloud_normals));
      normalfilter.filter(output);
    }
    
    Eigen::Vector3d PointcloudProc::projectPoint(Eigen::Vector4d& point, CamParams cam)
    {
      Eigen::Vector3d keypoint;
      
      keypoint(0) = (cam.fx*point.x()) / point.z() + cam.cx;
      keypoint(1) = (cam.fy*point.y()) / point.z() + cam.cy;
      keypoint(2) = (cam.fx*(point.x()-cam.tx)/point.z() + cam.cx);
      
      return keypoint;
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

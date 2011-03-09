/*#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <opencv2/features2d/features2d.hpp>

#include <frame_common/frame.h>
#include <boost/shared_ptr.hpp>
#include <cstdio>
#include <fstream>
#include <dirent.h>
#include <fnmatch.h>

#include <opencv/highgui.h>

using namespace std;
using namespace cv;
using namespace frame_common;*/
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

#include <vslam_system/vslam.h>
#include <posest/pe3d.h>
#include <sba/sba.h>
#include <sba/sba_file_io.h>
#include <frame_common/frame.h>
#include <boost/shared_ptr.hpp>
#include <cstdio>
#include <fstream>
#include <dirent.h>
#include <fnmatch.h>

#include <opencv/highgui.h>

using namespace std;
using namespace sba;
using namespace frame_common;
using namespace cv;
using namespace vslam;

// Names of left and right files in directory (with wildcards)
char *lreg, *rreg, *dreg;

// Filters for scandir
int getleft(struct dirent const *entry)
{
  if (!fnmatch(lreg, entry->d_name, 0))
    return 1;
  return 0;
}

int getright(struct dirent const *entry)
{
  if (!fnmatch(rreg, entry->d_name, 0))
    return 1;
  return 0;
}

int getidir(struct dirent const *entry)
{
  if (!fnmatch(dreg, entry->d_name, 0))
    return 1;
  return 0;
}

class CV_EXPORTS HowardDescriptorExtractor : public DescriptorExtractor
{
public:
    HowardDescriptorExtractor(int _neighborhoodSize = 7): neighborhoodSize(_neighborhoodSize)
    {
      CV_Assert(neighborhoodSize/2 != 0);
    }

    virtual void read(const FileNode &fn)
    {
      neighborhoodSize = fn["neighborhoodSize"];
    }
    virtual void write(FileStorage &fs) const
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
    virtual void computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
    {
      Mat im;
      if (image.channels() > 1)
      {
        cvtColor(image, im, CV_BGR2GRAY);
      }
      else
        im = image;
      int borderSize = neighborhoodSize/2;
      removeBorderKeypoints(keypoints, im.size(), borderSize);
      descriptors.create(keypoints.size(), descriptorSize(), descriptorType());

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
            descriptors.at<unsigned char>(keypointInd, index++) = im.at<unsigned char>(y, x);
          }
        }
      }
    }
};

class HowardStereoMatcher
{
public:
  HowardStereoMatcher(float thresh): threshold(thresh)
  {
  }

  void match(const frame_common::Frame& prevFrame, const frame_common::Frame& frame,
             vector<DMatch>& matches)
  {
    Mat scoreMatrix;
    calculateScoreMatrix(prevFrame, frame, scoreMatrix);
    calculateCrossCheckMatches(scoreMatrix, matches);
    Mat consistMatrix;
    calculateConsistMatrix(matches, prevFrame, frame, consistMatrix);
    filterMatches(consistMatrix, matches);
  }

private:
  void calculateScoreMatrix(const frame_common::Frame& prevFrame, const frame_common::Frame& frame, Mat& scoreMatrix)
  {
    scoreMatrix.create(prevFrame.dtors.rows, frame.dtors.rows, CV_32S);
    for (int row = 0; row < prevFrame.dtors.rows; row++)
      for (int col = 0; col < frame.dtors.rows; col++)
      {
        //calculate SAD between row descriptor from first image and col descriptor from second image
        int sad = 0;
        for (int i = 0; i < prevFrame.dtors.cols; i++)
        {
          sad += abs(prevFrame.dtors.at<unsigned char>(row, i) - frame.dtors.at<unsigned char>(col, i));
        }
        scoreMatrix.at<int>(row, col) = sad;
      }
  }

  void calculateCrossCheckMatches(const Mat& scoreMatrix, vector<DMatch>& matches)
  {
    cv::Point minIndex;
    vector<int> matches1to2(scoreMatrix.rows);
    for (int row = 0; row < scoreMatrix.rows; row++)
    {
      cv::minMaxLoc(scoreMatrix.row(row), 0, 0, &minIndex, 0, Mat());
      matches1to2[row] = minIndex.x;
    }
    vector<int> matches2to1(scoreMatrix.cols);
    for (int col = 0; col < scoreMatrix.cols; col++)
    {
      cv::minMaxLoc(scoreMatrix.col(col), 0, 0, &minIndex, 0, Mat());
      matches2to1[col] = minIndex.y;
    }

    for (size_t mIndex = 0; mIndex < matches1to2.size(); mIndex++)
    {
      if (matches2to1[matches1to2[mIndex]] == (int)mIndex)
      {
        DMatch match;
        match.trainIdx = mIndex;
        match.queryIdx = matches1to2[mIndex];
        matches.push_back(match);
      }
    }
  }


  void calculateConsistMatrix(const vector<DMatch>& matches, const frame_common::Frame& prevFrame,
                              const frame_common::Frame& frame, Mat& consistMatrix)
  {
    consistMatrix.create(matches.size(), matches.size(), CV_8UC1);
    for (int row = 0; row < consistMatrix.rows; row++)
    {
      if (!prevFrame.goodPts[matches[row].trainIdx])
      {
         Mat rowMatrix = consistMatrix.row(row);
         rowMatrix.setTo(Scalar(0));
      }
      else
      {
        for (int col = 0; col < row; col++)
        {
          unsigned char consistent = 0;
          if (frame.goodPts[matches[col].queryIdx])
          {
            Eigen::Vector4d vec = prevFrame.pts[matches[row].trainIdx];
            Point3f p11(vec(0), vec(1), vec(2));
            vec = prevFrame.pts[matches[col].trainIdx];
            Point3f p21(vec(0), vec(1), vec(2));

            vec = frame.pts[matches[row].queryIdx];
            Point3f p12(vec(0), vec(1), vec(2));
            vec = frame.pts[matches[col].queryIdx];
            Point3f p22(vec(0), vec(1), vec(2));

            Point3f diff1 = p11 - p21;
            Point3f diff2 = p12 - p22;
            if (abs(norm(diff1) - norm(diff2)) < threshold)
              consistent = 1;
          }
          consistMatrix.at<unsigned char>(row, col) = consistMatrix.at<unsigned char>(col, row) = consistent;
        }
      }
      consistMatrix.at<unsigned char>(row, row) = 1;
    }
  }

  void filterMatches(const Mat& consistMatrix, vector<DMatch>& matches)
  {
    vector<int> indices;
    //initialize clique
    Mat sums(1, consistMatrix.rows, CV_32S);
    for (int row = 0; row < consistMatrix.rows; row++)
      sums.at<int>(0, row) = cv::sum(consistMatrix.row(row))[0];
    cv::Point maxIndex;
    cv::minMaxLoc(sums, 0, 0, 0, &maxIndex, Mat());
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

    vector<DMatch> filteredMatches;
    for (size_t i = 0; i < indices.size(); i++)
      filteredMatches.push_back(matches[indices[i]]);
    matches = filteredMatches;
  }

  float threshold;
};

struct Pose
{
  Point3f center, x, y, z;
  Mat rvec, tvec;
};

void project3dPoint(const Point3f& point, const Mat& rvec, const Mat& tvec, Point3f& modif_point)
{
  Mat R(3, 3, CV_64FC1);
  Rodrigues(rvec, R);
  modif_point.x = R.at<double> (0, 0) * point.x + R.at<double> (0, 1) * point.y + R.at<double> (0, 2)
     * point.z + tvec.at<double> (0, 0);
  modif_point.y = R.at<double> (1, 0) * point.x + R.at<double> (1, 1) * point.y + R.at<double> (1, 2)
     * point.z + tvec.at<double> (1, 0);
  modif_point.z = R.at<double> (2, 0) * point.x + R.at<double> (2, 1) * point.y + R.at<double> (2, 2)
     * point.z + tvec.at<double> (2, 0);
}

// draw the graph on rviz
void stereodrawgraph(const SysSBA &sba, const ros::Publisher &cam_pub)
{
  visualization_msgs::Marker cammark;
  cammark.header.frame_id = "/pgraph";
  cammark.header.stamp = ros::Time();
  cammark.ns = "pgraph";
  cammark.id = 0;
  cammark.action = visualization_msgs::Marker::ADD;
  cammark.pose.position.x = 0;
  cammark.pose.position.y = 0;
  cammark.pose.position.z = 0;
  cammark.pose.orientation.x = 0.0;
  cammark.pose.orientation.y = 0.0;
  cammark.pose.orientation.z = 0.0;
  cammark.pose.orientation.w = 1.0;
  cammark.scale.x = 0.02;
  cammark.scale.y = 0.02;
  cammark.scale.z = 0.02;
  cammark.color.r = 1.0f;
  cammark.color.g = 0.0f;
  cammark.color.b = 1.0f;
  cammark.color.a = 1.0f;
  cammark.lifetime = ros::Duration();
  cammark.type = visualization_msgs::Marker::LINE_LIST;

  int npts = sba.tracks.size();

  cout << "Number of points to draw: " << npts << endl;
  if (npts <= 0) return;


  // draw cameras
  int ncams = sba.nodes.size();
  cammark.points.resize(ncams*6);
  for (int i=0, ii=0; i<ncams; i++)
    {
      const Node &nd = sba.nodes[i];
      Vector3d opt;
      Matrix<double,3,4> tr;
      transformF2W(tr,nd.trans,Quaternion<double>(nd.qrot));

      cammark.points[ii].x = nd.trans.x();
      cammark.points[ii].y = nd.trans.z();
      cammark.points[ii++].z = -nd.trans.y();
      opt = tr*Vector4d(0,0,0.3,1);
      cammark.points[ii].x = opt.x();
      cammark.points[ii].y = opt.z();
      cammark.points[ii++].z = -opt.y();

      cammark.points[ii].x = nd.trans.x();
      cammark.points[ii].y = nd.trans.z();
      cammark.points[ii++].z = -nd.trans.y();
      opt = tr*Vector4d(0.2,0,0,1);
      cammark.points[ii].x = opt.x();
      cammark.points[ii].y = opt.z();
      cammark.points[ii++].z = -opt.y();

      cammark.points[ii].x = nd.trans.x();
      cammark.points[ii].y = nd.trans.z();
      cammark.points[ii++].z = -nd.trans.y();
      opt = tr*Vector4d(0,0.1,0,1);
      cammark.points[ii].x = opt.x();
      cammark.points[ii].y = opt.z();
      cammark.points[ii++].z = -opt.y();
    }

  cam_pub.publish(cammark);
}

// draw the graph on rviz
void drawgraph(const ros::Publisher &cam_pub, const vector<Pose>& cameras)
{
  static int index = 0;
  visualization_msgs::Marker cammark;
  cammark.header.frame_id = "/pgraph";
  cammark.header.stamp = ros::Time();
  cammark.ns = "pgraph";
  cammark.id = index++;
  cammark.action = visualization_msgs::Marker::ADD;
  cammark.pose.position.x = 0;
  cammark.pose.position.y = 0;
  cammark.pose.position.z = 0;
  cammark.pose.orientation.x = 0.0;
  cammark.pose.orientation.y = 0.0;
  cammark.pose.orientation.z = 0.0;
  cammark.pose.orientation.w = 1.0;
  cammark.scale.x = 0.02;
  cammark.scale.y = 0.02;
  cammark.scale.z = 0.02;
  cammark.color.r = 0.0f;
  cammark.color.g = 1.0f;
  cammark.color.b = 1.0f;
  cammark.color.a = 1.0f;
  cammark.lifetime = ros::Duration();
  cammark.type = visualization_msgs::Marker::LINE_LIST;

  // draw cameras
  cammark.points.resize(6);

  const Pose& lastCamera = cameras[cameras.size()-1];
  cammark.points[0].x = cammark.points[2].x = cammark.points[4].x = lastCamera.center.x;
  cammark.points[0].y = cammark.points[2].y = cammark.points[4].y = lastCamera.center.y;
  cammark.points[0].z = cammark.points[2].z = cammark.points[4].z = lastCamera.center.z;
  cammark.points[1].x = lastCamera.x.x;
  cammark.points[1].y = lastCamera.x.y;
  cammark.points[1].z = lastCamera.x.z;
  cammark.points[3].x = lastCamera.y.x;
  cammark.points[3].y = lastCamera.y.y;
  cammark.points[3].z = lastCamera.y.z;
  cammark.points[5].x = lastCamera.z.x;
  cammark.points[5].y = lastCamera.z.y;
  cammark.points[5].z = lastCamera.z.z;

  cam_pub.publish(cammark);
}


int main(int argc, char** argv)
{
  if (argc < 5)
  {
    printf("Args are: <param file> <image dir> <left image file template> <right image file template>\n");
    //exit(0);
  }

  // get camera parameters, in the form: fx fy cx cy tx
  fstream fstr;
  fstr.open(argv[1], fstream::in);
  if (!fstr.is_open())
  {
    printf("Can't open camera file %s\n", argv[1]);
    exit(0);
  }
  CamParams camp;
  fstr >> camp.fx;
  fstr >> camp.fy;
  fstr >> camp.cx;
  fstr >> camp.cy;
  fstr >> camp.tx;
  Mat intrinsic = Mat::zeros(Size(3, 3), CV_64F);
  intrinsic.at<double>(0, 0) = camp.fx;
  intrinsic.at<double>(0, 2) = camp.cx;
  intrinsic.at<double>(1, 1) = camp.fy;
  intrinsic.at<double>(1, 2) = camp.cy;
  intrinsic.at<double>(2, 2) = 1.0;

  cout << "Cam params: " << camp.fx << " " << camp.fy << " " << camp.cx << " " << camp.cy << " " << camp.tx << endl;

  // set up directories
  struct dirent **lims, **rims, **dirs;
  int nlim, nrim, ndirs;
  string dname = argv[2];
  if (!dname.compare(dname.size() - 1, 1, "/")) // see if slash at end
    dname.erase(dname.size() - 1);

  string dirfn = dname.substr(dname.rfind("/") + 1);
  string tdir = dname.substr(0, dname.rfind("/") + 1);
  cout << "Top directory is " << tdir << endl;
  cout << "Search directory name is " << dirfn << endl;
  dreg = (char *)dirfn.c_str();

  ndirs = scandir(tdir.c_str(), &dirs, getidir, alphasort);
  printf("Found %d directories\n", ndirs);
  printf("%s\n", dirs[0]->d_name);

  ////////////////////////////////////
  // set up structures
  vslam::VslamSystem vslam(argv[5],argv[6]);
  typedef cv::CalonderDescriptorExtractor<float> Calonder;
  vslam.frame_processor_.setFrameDescriptor(new Calonder(argv[7]));

  // parameters
  vslam.setKeyDist(0.4);        // meters
  vslam.setKeyAngle(0.2);       // radians
  vslam.setKeyInliers(300);
  vslam.setHuber(2.0);          // Huber cost function cutoff
  //  vslam.vo_.pose_estimator_->wy = 64;
  //  vslam.vo_.pose_estimator_->wx = 64;
  //  vslam.vo_.pose_estimator_->numRansac = 1000;
  vslam.vo_.sba.verbose = false;
  vslam.sba_.verbose = false;
  ////////////////////////////////////

  // set up markers for visualization
  ros::init(argc, argv, "VisBundler");
  ros::NodeHandle nh("~");
  ros::Publisher cam_pub = nh.advertise<visualization_msgs::Marker> ("cameras", 0);
  ros::Publisher stereo_cam_pub = nh.advertise<visualization_msgs::Marker> ("stereo_cameras", 0);

  const std::string window_name = "matches";
  cv::namedWindow(window_name, 0);
  cv::Mat display;

  int iter = 0;
  lreg = argv[3];
  rreg = argv[4];

  // loop over directories
  for (int dd = 0; dd < ndirs; dd++)
  {
    char dir[2048];
    sprintf(dir, "%s%s", tdir.c_str(), dirs[dd]->d_name);
    printf("Current directory: %s\n", dir);

    // get left/right image file names, sorted
    nlim = scandir(dir, &lims, getleft, alphasort);
    printf("Found %d left images\n", nlim);
    printf("%s\n", lims[0]->d_name);

    nrim = scandir(dir, &rims, getright, alphasort);
    printf("Found %d right images\n", nrim);
    printf("%s\n", rims[0]->d_name);

    if (nlim != nrim)
    {
      printf("Number of left/right images does not match: %d vs. %d\n", nlim, nrim);
      exit(0);
    }

    frame_common::Frame prevFrame;
    frame_common::FrameProc frameProcessor(10);
    Ptr<DescriptorExtractor> extractor = new HowardDescriptorExtractor(17);
    Ptr<FeatureDetector> detector = new FastFeatureDetector(20, true);
    frameProcessor.setFrameDescriptor(extractor);
    frameProcessor.setFrameDetector(detector);
    HowardStereoMatcher matcher(0.01);
    vector<Pose> cameras;
    float len = 0.1;
    Pose pose;
    pose.center = Point3f(0, 0, 0);
    pose.x = Point3f(len, 0, 0);
    pose.y = Point3f(0, len, 0);
    pose.z = Point3f(0, 0, len);
    Mat Rotation = Mat::ones(Size(3, 3), CV_64F);
    Mat rvec, tvec(3, 1, CV_64F, 0.0);
    Rodrigues(Rotation, rvec);
    pose.rvec = rvec;
    pose.tvec = tvec;
    cameras.push_back(pose);

    bool matched = true;
    // loop over each stereo pair, adding it to the system
    for (int ii = 0; ii < nlim; iter++, ii++)
    {
      // Load images
      char fn[2048];
      sprintf(fn, "%s/%s", dir, lims[ii]->d_name);
      printf("%s\n", fn);
      cv::Mat image1 = cv::imread(fn, 0);
      sprintf(fn, "%s/%s", dir, rims[ii]->d_name);
      printf("%s\n", fn);
      cv::Mat image1r = cv::imread(fn, 0);
      if (image1.rows == 0 || image1r.rows == 0)
         exit(0);

      ////////////////////////////////////////////////////////////
      bool is_keyframe = vslam.addFrame(camp, image1, image1r);

      if (is_keyframe)
      {
          int n = vslam.sba_.nodes.size();
          int np = vslam.sba_.tracks.size();

          // draw graph
          cout << "drawing with " << n << " nodes and " << np << " points..." << endl << endl;
          if (n%2 == 0)
            stereodrawgraph(vslam.sba_, stereo_cam_pub);

          // write file out
          if (n > 10 && n%500 == 0)
          {
              char fn[1024];
              sprintf(fn,"newcollege%d.g2o", n);
              sba::writeGraphFile(fn,vslam.sba_);
              sprintf(fn,"newcollege%dm.g2o", n);
              sba::writeGraphFile(fn,vslam.sba_,true);
          }

          int nnsba = 10;
          if (n > 4 && n%nnsba == 0)
          {
            cout << "Running large SBA" << endl;
            vslam.refine();
          }
      }
      ////////////////////////////////////////////////////////////

      Mat im; image1.copyTo(im);
      //bilateralFilter(im, image1, -1, 1, 1);
      Mat imr; image1r.copyTo(imr);
      //bilateralFilter(imr, image1r, -1, 1, 1);

      frame_common::Frame frame;
      frame.setCamParams(camp);
      frame.frameId = ii;
      frameProcessor.setStereoFrame(frame, image1, image1r, 0, false);

      vector<DMatch> matches;
      vector<Point3f> opoints;
      vector<Point2f> ipoints;
      if (!prevFrame.img.empty())
      {
        matcher.match(prevFrame, frame, matches);

        display.create(prevFrame.img.rows, prevFrame.img.cols + frame.img.cols, CV_8UC1);
        cv::Mat left = display(cv::Rect(0, 0, prevFrame.img.cols, prevFrame.img.rows));
        prevFrame.img.copyTo(left);
        cv::Mat right = display(cv::Rect(prevFrame.img.cols, 0, frame.img.cols, frame.img.rows));
        frame.img.copyTo(right);
        for (size_t i = 0; i < prevFrame.kpts.size(); ++i)
        {
          cv::Point pt1 = prevFrame.kpts[i].pt;
          cv::circle(display, pt1, 3, Scalar(255));
        }
        for (size_t i = 0; i < frame.kpts.size(); ++i)
        {
          cv::Point pt1(frame.kpts[i].pt.x+prevFrame.img.cols,frame.kpts[i].pt.y);
          cv::circle(display, pt1, 3, Scalar(255));
        }

        cout << "Matches size = " << matches.size() << endl;
        for (size_t i = 0; i < matches.size(); ++i)
        {
          if (prevFrame.goodPts[matches[i].trainIdx] && frame.goodPts[matches[i].queryIdx])
          {
            cv::Point pt1(prevFrame.kpts[matches[i].trainIdx].pt.x,prevFrame.kpts[matches[i].trainIdx].pt.y);
            cv::Point pt2(frame.kpts[matches[i].queryIdx].pt.x+prevFrame.img.cols,frame.kpts[matches[i].queryIdx].pt.y);
            ipoints.push_back(frame.kpts[matches[i].queryIdx].pt);
            Eigen::Vector4d vec = prevFrame.pts[matches[i].trainIdx];
            Point3f op(vec(0), vec(1), vec(2));
            opoints.push_back(op);
            cv::line(display, pt1, pt2, Scalar(0, 255));
          }
        }

        cv::imshow(window_name, display);
        /*char key;
        while (true)
        {
          key = cv::waitKey(1);
          if (key == ' ')
            break;
          else if (key == 27)
            return -1;
        }*/
        cv::waitKey(2);
      }

      matched = (matches.size() > 10);

      if (prevFrame.img.empty() || matched)
      {
        if (matched)
        {
          Mat rvec, tvec;
          solvePnP(Mat(opoints), Mat(ipoints), intrinsic, Mat::zeros(Size(1, 5), CV_64F), rvec, tvec, false);

          vector<Point2f> projectedPoints;
          projectPoints(Mat(opoints), rvec, tvec, intrinsic, Mat::zeros(Size(1, 5), CV_64F), projectedPoints);
          float reprojectionError = 0;
          for (size_t pointInd = 0; pointInd < projectedPoints.size(); pointInd++)
          {
            float error = norm(projectedPoints[pointInd] - ipoints[pointInd]);
            if (error > reprojectionError)
              reprojectionError = error;
          }
          cout << "Reprojection error = " << reprojectionError << endl;
          if (reprojectionError > 1.3)
          {
            matched = false;
          }
          else
          {
            cout << rvec << endl << tvec << endl;
            Pose pose;
            const Pose& lastPose = cameras[cameras.size()-1];
            project3dPoint(lastPose.center, rvec, tvec, pose.center);
            project3dPoint(lastPose.x, rvec, tvec, pose.x);
            project3dPoint(lastPose.y, rvec, tvec, pose.y);
            project3dPoint(lastPose.z, rvec, tvec, pose.z);
            cameras.push_back(pose);
            drawgraph(cam_pub, cameras);
          }
        }

      }
      //if (matched || prevFrame.img.empty())
        prevFrame = frame;
      if (!nh.ok())
        return -1;
    }
  }
  return 0;
}

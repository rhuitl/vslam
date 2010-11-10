#ifndef ESM_HPP_
#define ESM_HPP_

#include <posest/MatrixFunctions>
namespace Eigen
{
using namespace Eigen3;
}

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <posest/lie_algebra.hpp>

class HomoESM
{
public:
  void setTemplateImage(const cv::Mat &image);
  void setTestImage(const cv::Mat &image);
  void track(int nIters, cv::Mat &H, double &rmsError, cv::Ptr<LieAlgebra> lieAlgebra = new LieAlgebraHomography()) const;
  void visualizeTracking(const cv::Mat &H, cv::Mat &visualization) const;
private:
  cv::Mat testImage;
  cv::Mat templateImage, templateImageRowDouble;
  cv::Mat templateDxRow, templateDyRow;
  cv::Mat templatePoints;
  cv::Mat xx, yy;

  vector<cv::Point2f> templateVertices;

  void computeJacobian(const cv::Mat &dx, const cv::Mat &dy, cv::Mat &J, cv::Ptr<LieAlgebra> lieAlgebra) const;
  static void computeGradient(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy);
  void constructImage(const cv::Mat &srcImage, const vector<cv::Point2f> &points, cv::Mat &intensity) const;
};

#endif /* ESM_HPP_ */

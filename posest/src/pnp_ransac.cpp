#include "posest/pnp_ransac.h"
#include <iostream>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/mutex.h"

using namespace std;
using namespace cv;
using namespace tbb;

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

void generateVar(vector<char>& mask, RNG& rng)
{
  size_t size = mask.size();
  for (size_t i = 0; i < size; i++)
  {
	int i1 = rng.uniform(0, size);
	int i2 = rng.uniform(0, size);
	char curr = mask[i1];
	mask[i1] = mask[i2];
	mask[i2] = curr;
  }
}

void pnpTask(const vector<char>& used_points_mask, const Mat& camera_matrix, const Mat& dist_coeffs,
		const vector<Point3f>& object_points, const vector<Point2f>& image_points, vector<int>& inliers,
		float max_dist, cv::Mat& rvec, cv::Mat& tvec, tbb::mutex& Mutex)
{
  vector<Point3f> model_object_points;
  vector<Point2f> model_image_points;
  for (size_t i = 0; i < used_points_mask.size(); i++)
  {
    if (used_points_mask[i])
    {
      model_image_points.push_back(image_points[i]);
      model_object_points.push_back(object_points[i]);
    }
  }

  Mat rvecl, tvecl;

  //filter same 3d points, hang in solvePnP
  double eps = 1e-10;
  int num_same_points = 0;
  for (int i = 0; i < MIN_POINTS_COUNT; i++)
    for (int j = i + 1; j < MIN_POINTS_COUNT; j++)
      if (norm(model_object_points[i] - model_object_points[j]) < eps)
        num_same_points++;
  if (num_same_points > 0)
    return;
  
  solvePnP(Mat(model_object_points), Mat(model_image_points), camera_matrix, dist_coeffs, rvecl, tvecl);

  vector<Point2f> projected_points;
  projected_points.resize(object_points.size());
  projectPoints(Mat(object_points), rvecl, tvecl, camera_matrix, dist_coeffs, projected_points);

  vector<int> inliers_indexes;
  for (size_t i = 0; i < object_points.size(); i++)
  {
    if (norm(image_points[i] - projected_points[i]) < max_dist)
    {
      inliers_indexes.push_back(i);
    }
  }

  if (inliers_indexes.size() > inliers.size())
  {
    mutex::scoped_lock lock;
	lock.acquire(Mutex);
    inliers.clear();
    inliers.resize(inliers_indexes.size());
    memcpy(&inliers[0], &inliers_indexes[0], sizeof(int) * inliers_indexes.size());
    rvecl.copyTo(rvec);
    tvecl.copyTo(tvec);
	lock.release();
  }
}
void Iterate(const vector<Point3f>& object_points, const vector<Point2f>& image_points,
		const Mat& camera_matrix, const Mat& dist_coeffs, Mat& rvecl, Mat& tvecl, const float max_dist, const int min_inlier_num,
		vector<int>* inliers, RNG& rng, tbb::mutex& Mutex)
{
	vector<char> used_points_mask(object_points.size(), 0);
	memset(&used_points_mask[0], 1, MIN_POINTS_COUNT );
	generateVar(used_points_mask, rng);
	pnpTask(used_points_mask, camera_matrix, dist_coeffs, object_points, image_points, *inliers, max_dist, rvecl, tvecl, Mutex);
	if ((int)inliers->size() > min_inlier_num)
		task::self().cancel_group_execution();
}

class Iterator
{
	const vector<Point3f>* object_points;
	const vector<Point2f>* image_points;
	const Mat* camera_matrix;
	const Mat* dist_coeffs;
	Mat* rvecl;
	Mat* tvecl;
	const float max_dist;
	const int min_inlier_num;
	vector<int>* inliers;
	RNG* rng;
	static mutex ResultsMutex;
public:
    void operator()( const blocked_range<size_t>& r ) const {
        for( size_t i=r.begin(); i!=r.end(); ++i )
        {
        	Iterate(*object_points, *image_points, *camera_matrix, *dist_coeffs, *rvecl, *tvecl, max_dist, min_inlier_num, inliers, *rng, ResultsMutex);
        }
    }
    Iterator(const vector<Point3f>* tobject_points, const vector<Point2f>* timage_points,
            const Mat* tcamera_matrix, const Mat* tdist_coeffs, Mat* trvecl, Mat* ttvecl, float tmax_dist, int tmin_inlier_num,
            vector<int>* tinliers, RNG* trng):
            	object_points(tobject_points), image_points(timage_points),
            	camera_matrix(tcamera_matrix), dist_coeffs(tdist_coeffs), rvecl(trvecl), tvecl(ttvecl),
            	max_dist(tmax_dist), min_inlier_num(tmin_inlier_num), inliers(tinliers), rng(trng)
    {}
};
mutex Iterator::ResultsMutex;

bool solvePnPRansac(const vector<Point3f>& object_points, const vector<Point2f>& image_points,
                    const Mat& camera_matrix, const Mat& dist_coeffs, Mat& rvec, Mat& tvec, bool use_extrinsic_guess, //don't used now, introduces for compatibility with solvePnP interface
                    int num_iterations, float max_dist, int min_inlier_num, vector<int>* inliers)
{
  assert (object_points.size() == image_points.size());

  if (object_points.size() < MIN_POINTS_COUNT)
    return false;

  rvec.create(3, 1, CV_64FC1);
  tvec.create(3, 1, CV_64FC1);

  if (min_inlier_num == -1)
    min_inlier_num = object_points.size();

  vector<int> local_inliers;
  if (!inliers)
  {
    inliers = &local_inliers;
  }

  RNG rng;
  Mat rvecl(3, 1, CV_64FC1), tvecl(3, 1, CV_64FC1);

  task_scheduler_init TBBinit;
  parallel_for(blocked_range<size_t>(0,num_iterations), Iterator(&object_points, &image_points, &camera_matrix, &dist_coeffs, &rvecl, &tvecl, max_dist, min_inlier_num, inliers, &rng));

  if ((int)(*inliers).size() >= MIN_POINTS_COUNT)
  {
    vector<Point3f> model_object_points;
    vector<Point2f> model_image_points;
    int index;
    for (size_t i = 0; i < (*inliers).size(); i++)
    {
      index = (*inliers)[i];
      model_image_points.push_back(image_points[index]);
      model_object_points.push_back(object_points[index]);
    }
    rvecl.copyTo(rvec);
    tvecl.copyTo(tvec);
    solvePnP(Mat(model_object_points), Mat(model_image_points), camera_matrix, dist_coeffs, rvec, tvec, true);
  }
  else
  {
    rvec.setTo(Scalar::all(0));
    tvec.setTo(Scalar::all(0));
  }
  return true;
}

#ifndef SBA_BUNDLER_FILE_H
#define SBA_BUNDLER_FILE_H

#ifndef EIGEN_USE_NEW_STDVECTOR
#define EIGEN_USE_NEW_STDVECTOR
#endif // EIGEN_USE_NEW_STDVECTOR

//#define EIGEN_DEFAULT_IO_FORMAT Eigen3::IOFormat(10)

#include <Eigen3/Eigen>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "sba/sba.h"

namespace Eigen3
{
  typedef Matrix<double,11,1> Vector11d;
  typedef Matrix<double,5,1>  Vector5d;
}

namespace sba
{

  /** \brief Reads bundle adjustment data from a Bundler file to an instance of SysSBA.
   *
   * \param filename The name of the bundler-formatted file to read from.
   * \param sbaout An instance of SBA that the file will be written to.
   * It should be noted that the bundler format does not support stereo points;
   * all points read in will be monocular.
   * Note: documentation of the Bundler format can be found at 
   * http://phototour.cs.washington.edu/bundler/bundler-v0.3-manual.html .
   */
  int readBundlerFile(const char *filename, sba::SysSBA& sbaout);

  /** \brief Writes bundle adjustment data from an instance of SysSBA to a Bundler file.
   *
   * \param filename The name of the bundler-formatted file to write to.
   * The file is created, or if it already exists, truncated.
   * \param sbain An instance of SBA that the file will be written from.
   * It should be noted that the bundler format does not support stereo points;
   * all points written will be monocular. Also, since SBA does not store point
   * color information, all points will be colored white.
   * Note: documentation of the Bundler format can be found at 
   * http://phototour.cs.washington.edu/bundler/bundler-v0.3-manual.html .
   */
  int writeBundlerFile(const char *filename, sba::SysSBA& sbain);

  /** \brief A low-level parser for bundler files. */
  int 
  ParseBundlerFile(const char *fin,	// input file
		  std::vector< Eigen3::Vector3d, Eigen3::aligned_allocator<Eigen3::Vector3d> > &camp, // cam params <f d1 d2>
		  std::vector< Eigen3::Matrix3d, Eigen3::aligned_allocator<Eigen3::Matrix3d> > &camR, // cam rotation matrix
		  std::vector< Eigen3::Vector3d, Eigen3::aligned_allocator<Eigen3::Vector3d> > &camt, // cam translation
		  std::vector< Eigen3::Vector3d, Eigen3::aligned_allocator<Eigen3::Vector3d> > &ptp, // point position
		  std::vector< Eigen3::Vector3i, Eigen3::aligned_allocator<Eigen3::Vector3i> > &ptc, // point color
		  std::vector< std::vector< Eigen3::Vector4d, Eigen3::aligned_allocator<Eigen3::Vector4d> > > &ptt // point tracks - each vector is <camera_index u v>
    );

  /** \brief Reads bundle adjustment data from a graph-type file to an instance of SysSBA.
   *
   * \param filename The name of the bundler-formatted file to read from.
   * \param sbaout An instance of SBA that the file will be written to.
   * Note: where is the documentation for this format?
   */
  int readGraphFile(const char *filename, sba::SysSBA& sbaout);

  /** \brief A low-level parser for graph files. */
  int 
  ParseGraphFile(const char *fin,	// input file
		  std::vector< Eigen3::Vector5d, Eigen3::aligned_allocator<Eigen3::Vector5d> > &camps, // cam params <fx fy cx cy>
		  std::vector< Eigen3::Vector4d, Eigen3::aligned_allocator<Eigen3::Vector4d> > &camqs, // cam rotation matrix
		  std::vector< Eigen3::Vector3d, Eigen3::aligned_allocator<Eigen3::Vector3d> > &camts, // cam translation
		  std::vector< Eigen3::Vector3d, Eigen3::aligned_allocator<Eigen3::Vector3d> > &ptps, // point position
		  std::vector< std::vector< Eigen3::Vector11d, Eigen3::aligned_allocator<Eigen3::Vector11d> > > &ptts // point tracks - each vector is <camera_index u v>
    );


  /// write out system in SBA form
  void writeLourakisFile(const char *fname, SysSBA& sba);
  void writeA(const char *fname, SysSBA& sba); // save precision matrix
  void writeSparseA(const char *fname, SysSBA& sba); // save precision matrix in CSPARSE format

  /**
   * \brief Writes out the current SBA system as an ascii graph file
   * suitable to be read in by the Freiburg HChol system.
   * <mono> is true if only monocular projections are desired
   */
  int writeGraphFile(const char *filename, SysSBA& sba, bool mono=false);


}; // namespace SBA

#endif

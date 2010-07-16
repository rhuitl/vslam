#ifndef SBA_BUNDLER_FILE_H
#define SBA_BUNDLER_FILE_H

#ifndef EIGEN_USE_NEW_STDVECTOR
#define EIGEN_USE_NEW_STDVECTOR
#endif // EIGEN_USE_NEW_STDVECTOR

//#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(10)

#include <Eigen/Eigen>
#include <vector>
#include <iostream>
#include <fstream>

#include "sba/sba.h"

/** \brief Reads bundle adjustment data from a Bundler file to an instance of SysSBA.
 *
 * \param filename The name of the bundler-formatted file to read from.
 * \param sbaout An instance of SBA that the file will be written to.
 * It should be noted that the bundler format does not support stereo points;
 * all points read in will be monocular.
 * Note: documentation of the Bundler format can be found at 
 * http://phototour.cs.washington.edu/bundler/bundler-v0.3-manual.html .
 */
int read_bundler_to_sba(char *filename, sba::SysSBA& sbaout);

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
int write_bundler_from_sba(char *filename, sba::SysSBA& sbain);

/** \brief A low-level parser for bundler files. */
int 
ReadBundlerFile(char *fin,	// input file
		std::vector< Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &camp, // cam params <f d1 d2>
		std::vector< Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > &camR, // cam rotation matrix
		std::vector< Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &camt, // cam translation
		std::vector< Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &ptp, // point position
		std::vector< Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > &ptc, // point color
		std::vector< std::vector< Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > > &ptt // point tracks - each vector is <camera_index u v>
  );

#endif

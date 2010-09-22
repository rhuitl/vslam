#ifndef VOCABULARY_TREE_FEATURE_ALLOCATOR_H
#define VOCABULARY_TREE_FEATURE_ALLOCATOR_H

#include <Eigen3/Core>
#include <Eigen3/StdVector>

namespace vt {

/// Meta-function to get the default allocator for a particular feature type.
template<class Feature>
struct DefaultAllocator
{
  typedef std::allocator<Feature> type;
};

// Specialization for Eigen3::Matrix types.
template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct DefaultAllocator< Eigen3::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> >
{
  typedef Eigen3::aligned_allocator<Eigen3::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> > type;
};

} //namespace vt

#endif

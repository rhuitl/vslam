#!/bin/bash
# shell script for making a source distribution of the SPA2D library

DISTDIR=spa2d-0.1

mkdir -p ${DISTDIR} ${DISTDIR}/src ${DISTDIR}/include/ ${DISTDIR}/include/sba

cp src/spa2d.cpp ${DISTDIR}/src
cp src/csparse.cpp ${DISTDIR}/src
cp include/sba/csparse.h ${DISTDIR}/include/sba
cp include/sba/spa2d.h ${DISTDIR}/include/sba
tar -cvzf ${DISTDIR}.tgz ${DISTDIR}


# Making a standalone SPA2D library

# Dependencies
# ============

# External libs
#    Eigen
#    CSparse 
#    CHOLMOD/SuiteSparse (optional, uses GPL license)


# Include file setup
# ==================

# sba/include/sba                 spa2d.h, csparse.h
# csparse/include/CSparse         csparse.h


# Compile flags
# =============
# DSIF (include delayed state filter)
# CHOLMOD (include CHOLMOD libraries)


# API
# ===

# // make an SPA object
# SysSPA2d SysSPA2d()  

# // add nodes and constraints
# int SysSPA2d::addNode2d(Vector3d &pos)
# bool SysSPA2d::addConstraint2d(int nd0, int nd1, Vector3d &mean, Matrix3d &prec)

# // run the optimizer
# int SysSPA2d::doSPA(int iter, double lambda=1.0e-4)

# // return cost and nodes
# double SysSPA2d::getCost()
# std::vector<Node2d,Eigen3::aligned_allocator<Node2d> > SysSPA2d::getNodes()



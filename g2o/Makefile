all: g2o

SOURCE_DIR = build/g2o
SVN_DIR = $(SOURCE_DIR)
TOP_DIR = $(PWD)

# tarball download
#TARBALL = build/g2o.tar.gz
#TARBALL_URL = http://pr.willowgarage.com/downloads/g2o.tar.gz
#TARBALL_URL = http://vault.willowgarage.com/wgdata1/vol1/g2o/g2o.tar.gz
#TARBALL_URL = file:///home/konolige/devel/g2o.tar.gz
#TARBALL_PATCH = makefile.patch
#MD5SUM_FILE = g2o.tar.gz.md5sum
#include $(shell rospack find mk)/download_unpack_build.mk

# svn download
SVN_URL =  https://svn.openslam.org/data/svn/g2o/trunk
#SVN_REVISION = -r233
#SVN_CMDLINE = svn --no-auth-cache --trust-server-cert --non-interactive
SVN_CMDLINE = svn --non-interactive --config-dir $(TOP_DIR)/svnconf
include $(shell rospack find mk)/svn_checkout.mk

#g2o: $(SOURCE_DIR)/unpacked
g2o: $(SOURCE_DIR)
	-rm -rf lib include bin
	mkdir -p $(SOURCE_DIR)/build
	mkdir -p lib
	mkdir -p bin
	mkdir -p include/g2o
	mkdir -p include/g2o/core
	mkdir -p include/g2o/solvers
	mkdir -p include/g2o/solvers/csparse
	mkdir -p include/g2o/solvers/dense
	mkdir -p include/g2o/solvers/cholmod
	mkdir -p include/g2o/solvers/pcg
	mkdir -p include/g2o/types
	mkdir -p include/g2o/types/icp
	mkdir -p include/g2o/types/sba
	mkdir -p include/g2o/types/sim3
	mkdir -p include/g2o/types/slam2d
	mkdir -p include/g2o/types/slam3d
	mkdir -p include/g2o/math_groups
	cd $(SOURCE_DIR)/build && cmake ../ && make
	cp $(SOURCE_DIR)/lib/*.so lib
	cp $(SOURCE_DIR)/bin/* bin
	cp $(SOURCE_DIR)/g2o/core/*.h include/g2o/core
	cp $(SOURCE_DIR)/g2o/solvers/csparse/*.h include/g2o/solvers/csparse
	cp $(SOURCE_DIR)/g2o/solvers/dense/*.h include/g2o/solvers/dense
	cp $(SOURCE_DIR)/g2o/solvers/cholmod/*.h include/g2o/solvers/cholmod
	cp $(SOURCE_DIR)/g2o/solvers/pcg/*.h include/g2o/solvers/pcg
	cp $(SOURCE_DIR)/g2o/types/icp/*.h include/g2o/types/icp
	cp $(SOURCE_DIR)/g2o/types/sba/*.h include/g2o/types/sba
	cp $(SOURCE_DIR)/g2o/types/sim3/*.h include/g2o/types/sim3
	cp $(SOURCE_DIR)/g2o/types/slam2d/*.h include/g2o/types/slam2d
	cp $(SOURCE_DIR)/g2o/types/slam3d/*.h include/g2o/types/slam3d
	cp $(SOURCE_DIR)/g2o/math_groups/*.h include/g2o/math_groups
	touch g2o

clean:
	rm -rf lib bin include $(SOURCE_DIR)
wipe: clean
	-rm -rf build g2o



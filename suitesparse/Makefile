all: SuiteSparse

TARBALL = build/SuiteSparse.tar.gz
TARBALL_URL = http://pr.willowgarage.com/downloads/SuiteSparse-3.4.0.tar.gz
TARBALL_PATCH = makefile.patch
SOURCE_DIR = build/SuiteSparse
MD5SUM_FILE = SuiteSparse.tar.gz.md5sum
include $(shell rospack find mk)/download_unpack_build.mk

SuiteSparse: $(SOURCE_DIR)/unpacked
	-rm -rf lib include
	mkdir -p lib
	mkdir -p include/SuiteSparse
	cd $(SOURCE_DIR) && make library
	cp $(SOURCE_DIR)/CHOLMOD/Lib/*.so lib
	cp $(SOURCE_DIR)/AMD/Lib/*.so lib
	cp $(SOURCE_DIR)/CAMD/Lib/*.so lib
	cp $(SOURCE_DIR)/COLAMD/Lib/*.so lib
	cp $(SOURCE_DIR)/CCOLAMD/Lib/*.so lib
	cp $(SOURCE_DIR)/CSparse/Lib/*.so lib
	cp $(SOURCE_DIR)/CHOLMOD/Include/*.h include/SuiteSparse
	cp $(SOURCE_DIR)/AMD/Include/*.h include/SuiteSparse
	cp $(SOURCE_DIR)/CAMD/Include/*.h include/SuiteSparse
	cp $(SOURCE_DIR)/COLAMD/Include/*.h include/SuiteSparse
	cp $(SOURCE_DIR)/CCOLAMD/Include/*.h include/SuiteSparse
	cp $(SOURCE_DIR)/CSparse/Include/*.h include/SuiteSparse
	cp $(SOURCE_DIR)/UFconfig/*.h include/SuiteSparse
	touch SuiteSparse
clean:
	rm -rf lib include $(SOURCE_DIR)
wipe: clean
	-rm -rf build SuiteSparse

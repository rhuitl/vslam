all: CSparse

TARBALL = build/CSparse.tar.gz
TARBALL_URL = http://pr.willowgarage.com/downloads/CSparse.tar.gz
TARBALL_PATCH = makefile.patch
SOURCE_DIR = build/CSparse
MD5SUM_FILE = CSparse.tar.gz.md5sum
include $(shell rospack find mk)/download_unpack_build.mk

CSparse: $(SOURCE_DIR)/unpacked
	-rm -rf lib include
	mkdir -p lib
	mkdir -p include/CSparse
	cd $(SOURCE_DIR) && make library
	cp $(SOURCE_DIR)/Lib/libcsparse.a lib
	cp $(SOURCE_DIR)/Include/cs.h include/CSparse
	touch CSparse
clean:
	rm -rf lib include $(SOURCE_DIR)
wipe: clean
	-rm -rf build CSparse

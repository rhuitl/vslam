VOCAB_DIR=`rospack find vocabulary_tree`
CALONDER_DIR=`rospack find calonder_descriptor`
bin/run_stereo data/cam-newcollege.txt '/u/prdata/vslam_data/NewCollege/FullData/StereoImages_1225720*' rect*left.pnm rect*right.pnm $VOCAB_DIR/holidays.tree $VOCAB_DIR/holidays.weights $CALONDER_DIR/current.rtc

VOCAB_DIR=`rospack find vocabulary_tree`
CALONDER_DIR=`rospack find vslam_system`/data
bin/run_howard_stereo data/cam-newcollege.txt '/home/alex/test/Stereo_Im*' rect*left.pnm rect*right.pnm $VOCAB_DIR/holidays.tree $VOCAB_DIR/holidays.weights $CALONDER_DIR/calonder.rtc

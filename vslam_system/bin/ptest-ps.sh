VOCAB_DIR=`rospack find vocabulary_tree`
CALONDER_DIR=`rospack find vslam_system`/data
bin/run_ps data/cam-ps.txt '/wg/wgdata1/vol1/PS/test' img*.png disp*.png $VOCAB_DIR/holidays.tree $VOCAB_DIR/holidays.weights $CALONDER_DIR/calonder.rtc

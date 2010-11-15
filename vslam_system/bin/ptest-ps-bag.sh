VOCAB_DIR=`rospack find vocabulary_tree`
CALONDER_DIR=`rospack find vslam_system`/data
bin/run_ps_bag $1 $VOCAB_DIR/holidays.tree $VOCAB_DIR/holidays.weights $CALONDER_DIR/calonder.rtc

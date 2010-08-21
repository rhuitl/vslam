PACKAGE='vslam_system'
import roslib; roslib.load_manifest(PACKAGE)

from dynamic_reconfigure.parameter_generator import *

def add_params(gen):
    detector_enum = gen.enum( [gen.const("FAST", str_t, "FAST", "FAST detector"),
                               gen.const("Harris", str_t, "Harris", "Harris detector"),
                               gen.const("Star", str_t, "Star", "Star detector"),
                               gen.const("SURF", str_t, "SURF", "SURF detector")],
                              "Enum to set the keypoint detector" )
    gen.add("detector", str_t, 0, "Keypoint detector", "FAST", edit_method = detector_enum)

    gen.add("fast_threshold", int_t, 0, "FAST detector threshold", 10, 1, 200)
    gen.add("fast_nonmax_suppression", bool_t, 0, "FAST detector nonmax suppression on/off", True)

    gen.add("harris_max_keypoints", int_t, 0, "Max keypoints returned", 300, 1, 1000)
    gen.add("harris_block_size", int_t, 0, "Size of averaging block", 3, 1, 11)
    gen.add("harris_min_distance", double_t, 0, "Minimum distance between returned corners", 1.0, 1.0, 20.0)
    gen.add("harris_quality_level", double_t, 0, "Minimal accepted quality", 0.01, 0.0, 1.0)
    gen.add("harris_k", double_t, 0, "Harris detector free parameter", 0.04, 0.0, 1.0)

    gen.add("star_max_size", int_t, 0, "Max feature size", 16, 4, 128)
    gen.add("star_response_threshold", int_t, 0, "Threshold to eliminate weak features", 30, 0, 200)
    gen.add("star_line_threshold_projected", int_t, 0, "Threshold to eliminate edges", 10, 0, 50)
    gen.add("star_line_threshold_binarized", int_t, 0, "Another threshold to eliminate edges", 8, 0, 50)
    gen.add("star_suppress_nonmax_size", int_t, 0, "Size of neighborhood for nonmax suppression", 5, 0, 11)

    gen.add("surf_hessian_threshold", double_t, 0, "Minimum hessian response", 1000, 0, 8000)
    gen.add("surf_octaves", int_t, 0, "Number of octaves", 3, 1, 10)
    gen.add("surf_octave_layers", int_t, 0, "Number of layers within each octave", 4, 1, 10)

    gen.add("grid_adapter", bool_t, 0, "Grid partitioning adapter on/off", False)
    gen.add("grid_max_keypoints", int_t, 0, "Max total keypoints", 300, 1, 1000)
    gen.add("grid_rows", int_t, 0, "Grid rows", 4, 1, 8)
    gen.add("grid_cols", int_t, 0, "Grid columns", 4, 1, 8)

    gen.add("pr_ransac_iterations", int_t, 0, "Place Recognizer RANSAC Iterations", 5000, 1, 10000)
    gen.add("pr_polish", bool_t, 0, "Place Recognizer Polish with SBA on/off", True)
    
    gen.add("vo_ransac_iterations", int_t, 0, "Visual Odometry RANSAC Iterations", 1000, 1, 10000)
    gen.add("vo_polish", bool_t, 0, "Visual Odometry Polish with SBA on/off", True)


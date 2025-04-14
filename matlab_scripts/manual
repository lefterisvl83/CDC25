Summary of the scripts and the overall procedure:
1) Training: compute PR scaling weights and distrubance feedback gains by running 'PR_scaling_disturbance_feedback_design.m'
2) Calibration & tightening: compute robustness tightening parameters by computing quantiles and Lipschitz constants of predicates by running 'Calibrate_quantile_compute_tightening_parameters.m' and 'Lipschitz_constants.m' 
3) Distributed implementation: Run 'distributed_STL_control_ten_agents_example.m', which uses inputs from above and utilizes the functions 'compute_10_agents_three_edges.m', 'compute_10_agents_two_edges.m' to solve local agent-level synthesis problem (see Alg. 2 in the paper) at every itearation.

Comparison with results in [1]: 'comparison_of_error_bounds_with_CDC24.m' returns Fig. 2 (left), that is comparison of the uncertainty quantification (for one agent) with the one in [1].

Fig. 1: A simulation of the distributed implementation proposed in Alg. 2 can be produced by directly running 'distributed_STL_control_ten_agents_example.m' in the same folder with 'compute_10_agents_three_edges.m', 'compute_10_agents_two_edges.m' and 'Lipschitz_constants.m'.

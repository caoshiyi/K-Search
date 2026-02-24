OE_CONFIG="<PATH_TO_KSEARCH_REPO>/baselines/openevolve/config_gqa_decode.yaml"
EVALUATOR_CONFIG="<PATH_TO_KSEARCH_REPO>/baselines/openevolve/flashinfer_evaluator_config_gqa_decode.yaml"
INITIAL_PROGRAM="<PATH_TO_KSEARCH_REPO>/baselines/openevolve/initial_kernel.txt"

export OPENAI_API_KEY="DUMMY_OPENAI_API_KEY"
export WANDB_API_KEY="DUMMY_WANDB_API_KEY"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" python3 -u "<PATH_TO_KSEARCH_REPO>/baselines/openevolve/run_evolve.py" \
	--config "$OE_CONFIG" \
	--evaluator-config "$EVALUATOR_CONFIG" \
	--initial-program "$INITIAL_PROGRAM" \
	--final-eval-all-workloads
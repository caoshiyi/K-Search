REPO_ROOT="<PATH_TO_KSEARCH_REPO>"
OE_CONFIG="${REPO_ROOT}/examples/openevolve/config.yaml"
EVALUATOR_CONFIG="${REPO_ROOT}/examples/openevolve/flashinfer_evaluator_config.yaml"
INITIAL_PROGRAM="${REPO_ROOT}/examples/openevolve/initial_empty.txt"

export OPENAI_API_KEY="DUMMY_OPENAI_API_KEY"

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 -u "${REPO_ROOT}/examples/openevolve/run_evolve.py" \
	--config "$OE_CONFIG" \
	--evaluator-config "$EVALUATOR_CONFIG" \
	--initial-program "$INITIAL_PROGRAM" \
	--final-eval-all-workloads

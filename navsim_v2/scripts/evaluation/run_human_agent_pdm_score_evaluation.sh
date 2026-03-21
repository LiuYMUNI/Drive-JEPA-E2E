CACHE_PATH='/home/linhan/yinlin/projects/navsim_workspace/exp/metric_cache_v2'
TRAIN_TEST_SPLIT=navtest

python -u $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=human_agent \
worker=single_machine_thread_pool \
worker.max_workers=16 \
worker.use_process_pool=true \
metric_cache_path=$CACHE_PATH \
experiment_name=human_agent_v2 

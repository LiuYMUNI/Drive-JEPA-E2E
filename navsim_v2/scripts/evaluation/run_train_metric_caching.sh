TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=/scratch/linhan/train_metric_cache_v2

python -u $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
worker=single_machine_thread_pool \
worker.max_workers=32 \
worker.use_process_pool=true \
metric_cache_path=$CACHE_PATH

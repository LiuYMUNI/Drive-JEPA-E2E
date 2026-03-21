TRAIN_TEST_SPLIT=navtrain

# torchrun --standalone --nproc_per_node=gpu $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_no_log.py \
torchrun --standalone --nproc_per_node=gpu $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=transfuser_agent \
        agent.config.latent=False \
        experiment_name=train_transfuser_agent_b32_e100_v2 \
        train_test_split=$TRAIN_TEST_SPLIT \
        trainer.params.max_epochs=100 \
        dataloader.params.batch_size=32 \
        cache_path="${NAVSIM_EXP_ROOT}/training_transfuser_cache/" \
        use_cache_without_dataset=True  \
        worker=single_machine_thread_pool  \
        force_cache_computation=False 
        # experiment_name=train_transfuser_agent_b32_e100 \

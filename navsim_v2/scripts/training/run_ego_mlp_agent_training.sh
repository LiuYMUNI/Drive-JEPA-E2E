python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=ego_status_mlp_agent \
        experiment_name=training_ego_mlp_agent_v2  \
        train_test_split=navtrain  \
        trainer.params.max_epochs=50 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        worker=single_machine_thread_pool  \
        force_cache_computation=False 

python3 ../networks/run_training.py --train_data ../data/simu/train --val_ratio 0.1 --test_ratio 0.1 --batch_size 64 --val_batch_size 5000 --test_batch_size 5000 --input_dim 5 --output_dim 3 --dropout 0.1 --model MLP_LRELU_d32_d32 --learning_rate 0.005 --timestamp_idx 0 --output ../models/MLP_test --tb_dir ../tensorboard/MLP --tb_log_name test-r0 --reader_mode classic --max_sequence_size 4 --trajectory_length 20 --max_iterations 30000
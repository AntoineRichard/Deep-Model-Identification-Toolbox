# MODEL SEARCH
network_list="/cs-share/dream/systemid_v2/scripts/network_list/RNN_networks_LRELU_real.txt"

# VALIDATION PARAMETERS
run_max=2

# TRAINING PARAMETERS
iterations=10000
used_points=12
pred_points=1
batch_size=64
batch_test_size=5000
batch_val_size=5000
drop_rate=0.1
learning_rate=0.001
traj_length=20

# PATH
root="/cs-share/dream/systemid_v2"
training_set="$root/data/real/kf_data_vrz_imu"
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/cs-share/dream/RSS-2020-Priorization"
to_file="$output_root/model_search_rnn_real.txt"
save_suffix="$output_root/results/GS/RNN_real"
tb_suffix="$output_root/tensorboard/GS/RNN_real"

# INIT
mkdir -p ${output_root}

while IFS= read -r model        
do
	for run in $(seq 0 $run_max)
	do
		mkdir -p ${save_suffix}/${model}/r${run}
		echo python3 ${python_script} --train_data ${training_set} --val_ratio 0.1 --test_ratio 0.1 --batch_size ${batch_size} --val_batch_size ${batch_val_size} --test_batch_size ${batch_test_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${learning_rate} --timestamp_idx 0 --output ${save_suffix}/${model}/r${run} --tb_dir ${tb_suffix} --tb_log_name ${model}-r${run} --reader_mode seq2seq --trajectory_length=${traj_length} --max_iterations ${iterations} --continuity_idx 1 >> ${to_file}
	done
done < "$network_list"

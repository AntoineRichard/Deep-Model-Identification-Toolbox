# GRID SEARCH PARAMETERS
superbatch_size_list="128 256 384 512 768 1024 1536 2048 4096 8192 16384"

# VALIDATION PARAMETERS
run_max=4

# TRAINING PARAMETERS
iterations=30000
used_points=4
pred_points=1
batch_size=64
batch_test_size=10000
batch_val_size=10000
drop_rate=0.1
model="MLP_LRELU_d32_d32"
learning_rate=0.005
traj_length=20

# PATH
root="/cs-share/dream/systemid_v2"
training_set="$root/data/simu/train-full-unbalanced"
validation_set="$root/data/simu/val"
test_set="$root/data/simu/test"
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/cs-share/dream/RSS-2020-Priorization"
to_file="$output_root/gridsearch_grad_full_unbalanced.txt"
save_suffix="$output_root/results/GS/GRAD_full_unbalanced"
tb_suffix="$output_root/tensorboard/GS/GRAD_full_unbalanced"

# INIT
mkdir -p ${output_root}

for superbatch_size in ${superbatch_size_list}
do
	for run in $(seq 0 $run_max)
	do
		mkdir -p ${save_suffix}/${superbatch_size}/r${run}
		echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --test_batch_size ${batch_test_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${learning_rate} --timestamp_idx 0 --output ${save_suffix}/${superbatch_size}/r${run} --tb_dir ${tb_suffix} --tb_log_name ${superbatch_size}-r${run} --reader_mode classic --priorization GRAD --max_sequence_size ${used_points} --trajectory_length ${traj_length} --superbatch_size ${superbatch_size} --max_iterations ${iterations} >> ${to_file}
	done
done


# GRID SEARCH PARAMETERS
hist_max=20
hist_min=1

# VALIDATION PARAMETERS
run_max=4

# TRAINING PARAMETERS
iterations="10000 20000 30000 40000 50000 60000 70000 80000 90000"
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
training_set="$root/data/simu/train-full-random"
validation_set="$root/data/simu/val"
test_set="$root/data/simu/test"
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/cs-share/dream/RSS-2020-Priorization"
to_file="$output_root/gridsearch_train_steps_full_random.txt"
save_suffix="$output_root/results/GS/TRAIN_STEPS_full_random"
tb_suffix="$output_root/tensorboard/GS/TRAIN_STEPS_full_random"

# INIT
mkdir -p ${output_root}

for it in ${iterations}
do
	for run in $(seq 0 $run_max)
	do
			mkdir -p ${save_suffix}/${it}/r${run}
			echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --test_batch_size ${batch_test_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${learning_rate} --timestamp_idx 0 --output ${save_suffix}/${it}/r${run} --tb_dir ${tb_suffix} --tb_log_name ${it}-r${run} --reader_mode classic --trajectory_length ${traj_length} --max_iterations ${it} --max_sequence_size ${used_points} >> ${to_file}
	done
done


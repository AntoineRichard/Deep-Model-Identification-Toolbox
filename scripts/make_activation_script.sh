#!/bin/bash
TANH_LIST=$1
SIGMOID_LIST=$2
LEAKY_RELU_LIST=$3

ROOT="/home/gpu_user/nvme-storage/ICAR-Benchmark/training_script"

output="/home/gpu_user/nvme-storage/ICAR-Benchmark/training_script/results"
DEST_FILE="/home/gpu_user/nvme-storage/ICAR-Benchmark/training_script/train_list_activation.txt"

DS_DRONE_TRAIN="/home/gpu_user/nvme-storage/ICAR-Benchmark/data/datasets/simu_drone/experiences/train"
DS_DRONE_TEST="/home/gpu_user/nvme-storage/ICAR-Benchmark/data/datasets/simu_drone/experiences/test"
DS_DRONE_VAL="/home/gpu_user/nvme-storage/ICAR-Benchmark/data/datasets/simu_drone/experiences/val"
DS_DRONE="${DS_DRONE_TRAIN} ${DS_DRONE_TEST} ${DS_DRONE_VAL}"
DS_ASCTEC="/home/gpu_user/nvme-storage/ICAR-Benchmark/data/datasets/real_drone_ethz"
DS_HERON_TRAIN="/home/gpu_user/nvme-storage/ICAR-Benchmark/data/datasets/simu_heron/experiences/rand-xl"
DS_HERON_TEST="/home/gpu_user/nvme-storage/ICAR-Benchmark/data/datasets/simu_heron/experiences/test"
DS_HERON_VAL="/home/gpu_user/nvme-storage/ICAR-Benchmark/data/datasets/simu_heron/experiences/val"
DS_HERON="${DS_HERON_TRAIN} ${DS_HERON_TEST} ${DS_HERON_VAL}"
test_iter=49
test_window=20
split=0
val=0
used_points=12
pred_points=1
iterations=40000
optimization="none"

mkdir "$output"

mkdir "$output/ASCTEC"
mkdir "$output/DRONE"
mkdir "$output/HERON"

mkdir "$output/ASCTEC/TANH"
mkdir "$output/DRONE/TANH"
mkdir "$output/HERON/TANH"

for architecture in $(cat $TANH_LIST)
  do
    for run in {0..4}
      do
        mkdir "$output/ASCTEC/TANH/${architecture}-r${run}"
        input_dim=13
        output_dim=7
        cmd_dim=6
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_ASCTEC} --output $output/ASCTEC/TANH/${architecture}-r${run}\
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type asctec" >> ${DEST_FILE} 
        mkdir "$output/DRONE/TANH/${architecture}-r${run}"
        input_dim=8
        output_dim=4
        cmd_dim=4
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_DRONE} --output $output/DRONE/TANH/${architecture}-r${run}\
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type drone" >> ${DEST_FILE} 
        mkdir "$output/HERON/TANH/${architecture}-r${run}"
        input_dim=5
        output_dim=3
        cmd_dim=2
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_HERON} --output $output/HERON/TANH/${architecture}-r${run}\
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type boat" >> ${DEST_FILE}
      done
  done


mkdir "$output/ASCTEC/SIGMOID"
mkdir "$output/DRONE/SIGMOID"
mkdir "$output/HERON/SIGMOID"

for architecture in $(cat $SIGMOID_LIST)
  do
    for run in {0..4}
      do
        mkdir "$output/ASCTEC/SIGMOID/${architecture}-r${run}"
        input_dim=13
        output_dim=7
        cmd_dim=6
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_ASCTEC} --output $output/ASCTEC/SIGMOID/${architecture}-r${run} \
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type asctec" >> ${DEST_FILE} 
        mkdir "$output/DRONE/SIGMOID/${architecture}-r${run}"
        input_dim=8
        output_dim=4
        cmd_dim=4
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_DRONE} --output $output/DRONE/SIGMOID/${architecture}-r${run}\
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type drone" >> ${DEST_FILE} 
        mkdir "$output/HERON/SIGMOID/${architecture}-r${run}"
        input_dim=5
        output_dim=3
        cmd_dim=2
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_HERON} --output $output/HERON/SIGMOID/${architecture}-r${run}\
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type boat" >> ${DEST_FILE} 
      done
  done

mkdir "$output/ASCTEC/LEAKY_RELU"
mkdir "$output/DRONE/LEAKY_RELU"
mkdir "$output/HERON/LEAKY_RELU"

for architecture in $(cat $LEAKY_RELU_LIST)
  do
    for run in {0..4}
      do
        mkdir "$output/ASCTEC/LEAKY_RELU/${architecture}-r${run}"
        input_dim=13
        output_dim=7
        cmd_dim=6
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_ASCTEC} --output $output/ASCTEC/LEAKY_RELU/${architecture}-r${run} \
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type asctec" >> ${DEST_FILE} 
        mkdir "$output/DRONE/LEAKY_RELU/${architecture}-r${run}"
        input_dim=8
        output_dim=4
        cmd_dim=4
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_DRONE} --output $output/DRONE/LEAKY_RELU/${architecture}-r${run}\
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type drone" >> ${DEST_FILE} 
        mkdir "$output/HERON/LEAKY_RELU/${architecture}-r${run}"
        input_dim=5
        output_dim=3
        cmd_dim=2
        echo "python3 ${ROOT}/training_main.py --dataset ${DS_HERON} --output $output/HERON/LEAKY_RELU/${architecture}-r${run}\
                                       --iterations ${iterations} --used_points ${used_points} --pred_points ${pred_points}\
                                       --test_iter ${test_iter} --test_window ${test_window} --split_test ${split}\
                                       --split_val ${val} --model_type ${architecture} --optimisation_type ${optimization}\
                                       --input_state_dim ${input_dim} --output_state_dim ${output_dim} --cmd_dim ${cmd_dim}\
                                       --ds_type boat" >> ${DEST_FILE} 
      done
  done

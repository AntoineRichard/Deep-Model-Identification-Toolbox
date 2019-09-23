#!/bin/bash
RNN_LIST=$1
LSTM_LIST=$2
GRU_LIST=$3

ROOT="/cs-share/dream/RSS-Kingfisher/lstm_scripts"

output="/cs-share/dream/RSS-Kingfisher/training_script/results"
DEST_FILE="/cs-share/dream/RSS-Kingfisher/training_script/train_list_rnn.txt"

DS_DRONE_TRAIN="/cs-share/dream/RSS-Kingfisher/data/datasets/simu_drone/experiences/train"
DS_DRONE_TEST="/cs-share/dream/RSS-Kingfisher/data/datasets/simu_drone/experiences/test"
DS_DRONE_VAL="/cs-share/dream/RSS-Kingfisher/data/datasets/simu_drone/experiences/val"
DS_DRONE="--train_data ${DS_DRONE_TRAIN} --test_data ${DS_DRONE_TEST} --val_data ${DS_DRONE_VAL}"
DS_ASCTEC="/cs-share/dream/RSS-Kingfisher/data/datasets/real_drone_ethz"
DS_HERON_TRAIN="/cs-share/dream/RSS-Kingfisher/data/datasets/simu_heron/experiences/rand-xl"
DS_HERON_TEST="/cs-share/dream/RSS-Kingfisher/data/datasets/simu_heron/experiences/test"
DS_HERON_VAL="/cs-share/dream/RSS-Kingfisher/data/datasets/simu_heron/experiences/val"
DS_HERON="--train_data ${DS_HERON_TRAIN} --test_data ${DS_HERON_TEST} --val_data ${DS_HERON_VAL}"
test_iter=49
test_window=20
split=0
val=0
sequence=16
epochs=100
optimization="none"

mkdir "$output"

mkdir "$output/ASCTEC"
mkdir "$output/DRONE"
mkdir "$output/HERON"

#mkdir "$output/ASCTEC/RNN"
mkdir "$output/DRONE/RNN"
mkdir "$output/HERON/RNN"

for architecture in $(cat $RNN_LIST)
  do
    state=$(echo ${architecture} | cut -c 7-8)
    layers=$(echo ${architecture} | cut -c 11)
    dense=$(echo ${architecture} | cut -c 14)
    if [ $dense = "1" ]
    then
      model="rnn_hsX_lX_m1"
    else
      dense_size=$(echo ${architecture} | cut -c 16-17)
      model="rnn_hsX_lX_m2_${dense_size}"
    fi
    #echo "state: ${state}, layers: ${layers}, dense: ${dense}, model: ${model}."
    for run in {0..4}
      do
        #mkdir "$output/ASCTEC/RNN/${architecture}-r${run}"
        #input_dim=13
        #output_dim=7
        #echo "python3 ${ROOT}/train.py --dataset ${DS_ASCTEC} --output $output/ASCTEC/RNN/${architecture}-r${run}\
        #      --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
        #      --output_dim ${output_dim} --rnn_model lstm_hsX_lX_m2_16 --max_sequence ${sequence}\
        #      --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
        #      --state_size ${state}" >> ${DEST_FILE}
        mkdir "$output/DRONE/RNN/${architecture}-r${run}"
        input_dim=8
        output_dim=4
        echo "python3 ${ROOT}/train.py ${DS_DRONE} --output $output/DRONE/RNN/${architecture}-r${run}\
              --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
              --output_dim ${output_dim} --rnn_model ${model} --max_sequence ${sequence}  --tb_dir ${output}/tensorboard/DRONE/RNN\
              --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
              --state_size ${state}" >> ${DEST_FILE}
        mkdir "$output/HERON/RNN/${architecture}-r${run}"
        input_dim=5
        output_dim=3
        echo "python3 ${ROOT}/train.py ${DS_HERON} --output $output/HERON/RNN/${architecture}-r${run}\
              --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
              --output_dim ${output_dim} --rnn_model ${model} --max_sequence ${sequence} --tb_dir ${output}/tensorboard/HERON/RNN\
              --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
              --state_size ${state}" >> ${DEST_FILE}
      done
  done

#mkdir "$output/ASCTEC/GRU"
mkdir "$output/DRONE/GRU"
mkdir "$output/HERON/GRU"
for architecture in $(cat $GRU_LIST)
  do
    state=$(echo ${architecture} | cut -c 7-8)
    layers=$(echo ${architecture} | cut -c 11)
    dense=$(echo ${architecture} | cut -c 14)
    if [ $dense = "1" ]
    then
      model="gru_hsX_lX_m1"
    else
      dense_size=$(echo ${architecture} | cut -c 16-17)
      model="gru_hsX_lX_m2_${dense_size}"
    fi
    #echo "state: ${state}, layers: ${layers}, dense: ${dense}, model: ${model}."
    for run in {0..4}
      do
        #mkdir "$output/ASCTEC/GRU/${architecture}-r${run}"
        #input_dim=13
        #output_dim=7
        #echo "python3 ${ROOT}/train.py --dataset ${DS_ASCTEC} --output $output/ASCTEC/GRU/${architecture}-r${run}\
        #      --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
        #      --output_dim ${output_dim} --rnn_model lstm_hsX_lX_m2_16 --max_sequence ${sequence}\
        #      --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
        #      --state_size ${state}" >> ${DEST_FILE}
        mkdir "$output/DRONE/GRU/${architecture}-r${run}"
        input_dim=8
        output_dim=4
        echo "python3 ${ROOT}/train.py ${DS_DRONE} --output $output/DRONE/GRU/${architecture}-r${run}\
              --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
              --output_dim ${output_dim} --rnn_model ${model} --max_sequence ${sequence}  --tb_dir ${output}/tensorboard/DRONE/GRU\
              --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
              --state_size ${state}" >> ${DEST_FILE}
        mkdir "$output/HERON/GRU/${architecture}-r${run}"
        input_dim=5
        output_dim=3
        echo "python3 ${ROOT}/train.py ${DS_HERON} --output $output/HERON/GRU/${architecture}-r${run}\
              --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
              --output_dim ${output_dim} --rnn_model ${model} --max_sequence ${sequence} --tb_dir ${output}/tensorboard/HERON/GRU\
              --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
              --state_size ${state}" >> ${DEST_FILE}
      done
  done

#mkdir "$output/ASCTEC/LSTM"
mkdir "$output/DRONE/LSTM"
mkdir "$output/HERON/LSTM"
for architecture in $(cat $LSTM_LIST)
  do
    state=$(echo ${architecture} | cut -c 8-9)
    layers=$(echo ${architecture} | cut -c 12)
    dense=$(echo ${architecture} | cut -c 15)
    if [ $dense = "1" ]
    then
      model="lstm_hsX_lX_m1"
    else
      dense_size=$(echo ${architecture} | cut -c 17-18)
      model="lstm_hsX_lX_m2_${dense_size}"
    fi
    #echo "state: ${state}, layers: ${layers}, dense: ${dense}, model: ${model}."
    for run in {0..4}
      do
        #mkdir "$output/ASCTEC/LSTM/${architecture}-r${run}"
        #input_dim=13
        #output_dim=7
        #echo "python3 ${ROOT}/train.py --dataset ${DS_ASCTEC} --output $output/ASCTEC/LSTM/${architecture}-r${run}\
        #      --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
        #      --output_dim ${output_dim} --rnn_model lstm_hsX_lX_m2_16 --max_sequence ${sequence}\
        #      --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
        #      --state_size ${state}" >> ${DEST_FILE}
        mkdir "$output/DRONE/LSTM/${architecture}-r${run}"
        input_dim=8
        output_dim=4
        echo "python3 ${ROOT}/train.py ${DS_DRONE} --output $output/DRONE/LSTM/${architecture}-r${run}\
              --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
              --output_dim ${output_dim} --rnn_model ${model} --max_sequence ${sequence}  --tb_dir ${output}/tensorboard/DRONE/LSTM\
              --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
              --state_size ${state}" >> ${DEST_FILE}
        mkdir "$output/HERON/LSTM/${architecture}-r${run}"
        input_dim=5
        output_dim=3
        echo "python3 ${ROOT}/train.py ${DS_HERON} --output $output/HERON/LSTM/${architecture}-r${run}\
              --optimization none --tb_log_name ${architecture}-r${run} --learning_rate 0.005 --input_dim ${input_dim}\
              --output_dim ${output_dim} --rnn_model ${model} --max_sequence ${sequence} --tb_dir ${output}/tensorboard/HERON/LSTM\
              --window_size ${test_window} --max_trajectories ${test_iter} --max_epoch ${epochs} --layer_number ${layers}\
              --state_size ${state}" >> ${DEST_FILE}
      done
  done

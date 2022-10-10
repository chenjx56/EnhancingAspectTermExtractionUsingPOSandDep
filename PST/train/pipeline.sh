#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"

# MODEL_PATH=model/bert-base-uncased
MODEL_PATH=model/pt-bert-base
TASK_DATA_PATH=../data/14lap
DATASETS=pt14lap
for seed in 111
do
echo "#######################   Only Train Data Without Source Data   #######################"
python3 -B run_sequence_label.py --do_train --do_eval --seed $seed --datasets ${DATASETS} \
        --pretrained_params ${MODEL_PATH} --output_dir ${TASK_DATA_PATH}/train \
        --train_batch_size 32 --eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 10 --eval_logging_steps 200 \
        --task_name aspect --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --train_max_seq_length 128 --eval_max_seq_length 128 --gradient_accumulation_steps 1 --overwrite_output_dir --schedule WarmupLinearSchedule \
        --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json --evaluate_during_training --save_steps 5000 --logging_global_step 100 

python3 -B process.py --sent2discriminate ${TASK_DATA_PATH}/train
python3 -B process.py --sent2discriminate ${TASK_DATA_PATH}/test

echo "####################### discriminate #######################"
python3 -B run_classification.py --do_train --do_eval --seed $seed \
        --pretrained_params ${MODEL_PATH} --output_dir ${TASK_DATA_PATH}/train/discriminate \
        --train_batch_size 32 --eval_batch_size 64 --learning_rate 3e-5 --num_train_epochs 8 --eval_logging_steps 500 \
        --task_name discriminate --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --train_max_seq_length 128 --eval_max_seq_length 128 --gradient_accumulation_steps 1 --overwrite_output_dir --schedule WarmupLinearSchedule \
        --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json --evaluate_during_training --save_steps 5000 --logging_global_step 100 



declare -A Round1=( ["preRound"]="train" ["curRound"]="train1000" ["start_index"]=0 ["end_index"]=1000 
                    ["eval_file_name"]="eval_train1000_label.txt" ["number"]="1000"
                    ["onlyTrain"]="train1000" ["joinTrain"]="train train1000")

declare -A Round2=( ["preRound"]="train1000" ["curRound"]="train2000" ["start_index"]=1000 ["end_index"]=3000 
                    ["eval_file_name"]="eval_train2000_label.txt" ["number"]="2000" 
                    ["onlyTrain"]="train1000 train2000" ["joinTrain"]="train train1000 train2000")

declare -A Round3=( ["preRound"]="train2000" ["curRound"]="train3000" ["start_index"]=3000 ["end_index"]=6000 
                    ["eval_file_name"]="eval_train3000_label.txt" ["number"]="3000" 
                    ["onlyTrain"]="train1000 train2000 train3000" ["joinTrain"]="train train1000 train2000 train3000")

declare -A Round4=( ["preRound"]="train3000" ["curRound"]="train4000" ["start_index"]=6000 ["end_index"]=10000 
                    ["eval_file_name"]="eval_train4000_label.txt" ["number"]="4000" 
                    ["onlyTrain"]="train1000 train2000 train3000 train4000" ["joinTrain"]="train train1000 train2000 train3000 train4000")

# declare -A Round5=( ["preRound"]="train4000" ["curRound"]="train5000" ["start_index"]=10000 ["end_index"]=15000 
#                     ["eval_file_name"]="eval_train5000_label.txt" ["number"]="5000" 
#                     ["onlyTrain"]="train1000 train2000 train3000 train4000 train5000" ["joinTrain"]="train train1000 train2000 train3000 train4000 train5000")

declare -a Round=("Round1" "Round2" "Round3" "Round4")

for item in "${Round[@]}"; do
    echo "####################### ${item} #######################:"
    declare -n dict="$item"  # now p is a reference to a variable "$member"

    echo "####################### Test Source Data   #######################"
    python3 -B run_sequence_label.py --do_eval --seed $seed --datasets ${DATASETS} --pretrained_params ${TASK_DATA_PATH}/${dict["preRound"]} --output_dir ${TASK_DATA_PATH}/${dict["curRound"]} \
            --start_index ${dict["start_index"]} --end_index ${dict["end_index"]} --eval_test_file_name ${dict["eval_file_name"]} --eval_batch_size 32 \
            --task_name aspect --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --eval_max_seq_length 128 --overwrite_output_dir \
            --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json 

    python3 -B process.py --evalResult2discri ${TASK_DATA_PATH}/${dict["curRound"]} --eval_file ${dict["eval_file_name"]}

    echo "####################### Discriminate Source Data Result  #######################"
    python3 -B run_classification.py --do_eval --seed $seed --pretrained_params ${TASK_DATA_PATH}/${dict["preRound"]}/discriminate --output_dir ${TASK_DATA_PATH}/${dict["curRound"]} \
            --eval_directory_name ${dict["curRound"]} --eval_batch_size 32 \
            --task_name discriminate --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --eval_max_seq_length 128 \
            --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json

    python3 -B process.py --discri2sent ${TASK_DATA_PATH}/${dict["curRound"]} --eval_file ${dict["eval_file_name"]} --pretrained_params ${MODEL_PATH}
    python3 -B process.py --sent2discriminate ${TASK_DATA_PATH}/${dict["curRound"]}

    echo "####################### Aspect Train Data With Source Data  (Only) (${dict["number"]}) #######################"
    python3 -B run_sequence_label.py --do_train --do_eval --seed $seed --datasets ${DATASETS} \
            --pretrained_params ${MODEL_PATH} --output_dir ${TASK_DATA_PATH}/${dict["curRound"]}/aspect \
            --trainList "${dict["onlyTrain"]}" --train_batch_size 32 --eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 8 --eval_logging_steps 200 \
            --task_name aspect --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --train_max_seq_length 128 --eval_max_seq_length 128 --gradient_accumulation_steps 1 --overwrite_output_dir --schedule WarmupLinearSchedule \
            --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json --evaluate_during_training --save_steps 5000 --logging_global_step 100 

    # echo "####################### Aspect Train Data With Source Data  (Join) (${dict["number"]}) #######################"
    # python3 -B run_sequence_label.py --do_train --do_eval \
    #         --pretrained_params ${MODEL_PATH} --output_dir ${TASK_DATA_PATH}/${dict["curRound"]} \
    #         --trainList "${dict["joinTrain"]}" --train_batch_size 32 --eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 5 --eval_logging_steps 200 \
    #         --task_name aspect --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --train_max_seq_length 128 --eval_max_seq_length 128 --gradient_accumulation_steps 1 --overwrite_output_dir --schedule WarmupLinearSchedule \
    #         --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json --evaluate_during_training --save_steps 5000 --logging_global_step 100 

    echo "####################### Aspect Train Data With Source Data  (Fune) (${dict["number"]}) #######################"
    python3 -B run_sequence_label.py --do_train --do_eval --seed $seed --datasets ${DATASETS} \
            --pretrained_params ${TASK_DATA_PATH}/${dict["curRound"]}/aspect --output_dir ${TASK_DATA_PATH}/${dict["curRound"]} \
            --train_batch_size 32 --eval_batch_size 32 --learning_rate 5e-5 --num_train_epochs 8 --eval_logging_steps 200 \
            --task_name aspect --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --train_max_seq_length 128 --eval_max_seq_length 128 --gradient_accumulation_steps 1 --overwrite_output_dir --schedule WarmupLinearSchedule \
            --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json --evaluate_during_training --save_steps 5000 --logging_global_step 100 

    echo "####################### Discriminate Train Data With Source Data  (Join) #######################"
    python3 -B run_classification.py --do_train --do_eval --seed $seed \
            --pretrained_params ${MODEL_PATH} --output_dir ${TASK_DATA_PATH}/${dict["curRound"]}/discriminate \
            --trainList "${dict["joinTrain"]}" --train_batch_size 32 --eval_batch_size 32 --learning_rate 3e-5 --num_train_epochs 8 --eval_logging_steps 500 \
            --task_name discriminate --do_lower_case --data_dir ${TASK_DATA_PATH} --bert_model bert --train_max_seq_length 128 --eval_max_seq_length 128 --gradient_accumulation_steps 1 --overwrite_output_dir --schedule WarmupLinearSchedule \
            --pretrained_vocab ${MODEL_PATH}/vocab.txt --pretrained_config ${MODEL_PATH}/config.json --evaluate_during_training --save_steps 5000 --logging_global_step 100 
done
done
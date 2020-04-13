# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

NAME_DATA="cuhk03labeled"
NAME_MODEL="resnet50_rga"
BATCH_SIZE=64
NUM_WORKER=8
NUM_FEATURE=2048
NUM_EPOCHS=600
NUM_GPU=2
SEED=16
START_SAVE=320
BRANCH_NAME="rgasc"

DATA_DIR="/YOUR_DATASET_PATH"
LOG_DIR="./logs/RGA-SC/cuhk03labeled_b64f2048"
LOG_FILE="${LOG_DIR}/train_log.txt"

if [ -d ${LOG_DIR} ]; then
	rm -r ${LOG_DIR}
fi

if [ ! -d ${LOG_DIR} ]; then
	echo ${LOG_DIR}" not exists!!!"
	mkdir -p ${LOG_DIR}
fi

echo "Begin to train."
CUDA_VISIBLE_DEVICES=4,5 python main_imgreid.py \
	-a ${NAME_MODEL} \
	-b ${BATCH_SIZE} \
	-d ${NAME_DATA} \
	-j ${NUM_WORKER} \
	--opt adam \
	--dropout 0 \
	--combine-trainval \
	--seed ${SEED} \
	--num_gpu ${NUM_GPU} \
	--epochs ${NUM_EPOCHS} \
	--features ${NUM_FEATURE} \
	--start_save ${START_SAVE} \
	--branch_name ${BRANCH_NAME} \
	--data-dir ${DATA_DIR} \
	--logs-dir ${LOG_DIR} \
	> ${LOG_FILE} 2>&1

echo "Begin to test."
WEIGHT_FILE="${LOG_DIR}/checkpoint_600.pth.tar"
LOG_FILE_TEST="${LOG_DIR}/eval_600.txt"
CUDA_VISIBLE_DEVICES=4,5 python main_imgreid.py \
	-a ${NAME_MODEL} \
	-b ${BATCH_SIZE} \
	-d ${NAME_DATA} \
	-j ${NUM_WORKER} \
	--opt adam \
	--dropout 0 \
	--combine-trainval \
	--seed ${SEED} \
	--num_gpu ${NUM_GPU} \
	--epochs ${NUM_EPOCHS} \
	--features ${NUM_FEATURE} \
	--start_save ${START_SAVE} \
	--branch_name ${BRANCH_NAME} \
	--data-dir ${DATA_DIR} \
	--logs-dir ${LOG_DIR} \
	--evaluate \
	--resume ${WEIGHT_FILE} \
	> ${LOG_FILE_TEST} 2>&1
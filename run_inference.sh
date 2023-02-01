SUBMIT_PATH="submit_utils"
if [ -n "$1" ]; then REPEATS=$1; else REPEATS=1; fi

WEIGHT_DECAY_ARRAY=(0.1)

rm -f tmp_logs/*
rm -f $SUBMIT_PATH/inf_args.txt
# rm -f $SUBMIT_PATH/inf_submit.sub

function populate_template {
export VAE_MODEL_PATH=$1
export SAVE_PATH="run$2"
export DATA_REPETITIONS="5"
export PROPERTIES_TO_TEST="" #"contents_binary_label"
export ACTIONS_TO_USE=""
export END_EPOCH="200"
export WEIGHT_DECAY="$3"

mkdir -p $1/inference/run$2
envsubst < $SUBMIT_PATH/infconfig.yaml > $1/inference/run$2/inf_config.yaml
}

function build_args_txt {
echo "-c $1/inference/run$2/inf_config.yaml">> $SUBMIT_PATH/inf_args.txt
}

CNT=1
for dir in runs_all_obj/*/ ; do
    DIR=$(realpath -s $dir)
    for rep in $(seq $REPEATS); do
        for wgt in "${WEIGHT_DECAY_ARRAY[@]}"; do
            populate_template $DIR $CNT $wgt
            build_args_txt $DIR $CNT
            CNT=$[CNT+1]
        done
    done
    #break 1
    #echo "$dir"
done


if [ -n "$2" ]
then
    condor_submit $2 "$SUBMIT_PATH"/inf_submit.sub
else
    python inference_main.py -c $(realpath -s $DIR)/inf_config.yaml --save_path=run$rep
fi






#unstable_runs = 88,89,110,116,140,145,159






# function build_submission_file {
# cat >$SUBMIT_PATH/inf_submit.sub <<EOL
# executable = /home/richardson/miniconda3/envs/ball_env/bin/python
# arguments = inference_main.py \$(a1)
# error = /home/richardson/robot_haptic_perception/class_tmp_logs/lin-6-$(Process).err
# output = /home/richardson/robot_haptic_perception/class_tmp_logs/lin-6-$(Process).out
# log = /home/richardson/robot_haptic_perception/class_tmp_logs/lin-6-$(Process).log
# request_memory = 20000
# request_cpus = 1
# request_gpus = 1
# requirements = (TARGET.CUDACapability >= 6.0) && (TARGET.CUDACapability <= 7.5) && (TARGET.CUDAGlobalMemoryMb > 10000)
# +MaxRunningPrice=100
# +RunningPriceExceededAction = "restart"
# queue a1 from $SUBMIT_PATH/inf_args.txt
# EOL
# cat $SUBMIT_PATH/inf_submit.sub
# } &> /dev/null

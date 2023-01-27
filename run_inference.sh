SUBMIT_PATH="submit_utils"




rm -f tmp_logs/*
rm -f $SUBMIT_PATH/inf_args.txt
# rm -f $SUBMIT_PATH/inf_submit.sub

function populate_template {
export VAE_MODEL_PATH=$1
export SAVE_PATH=""
export DATA_REPETITIONS="1"
export PROPERTIES_TO_TEST=""
export ACTIONS_TO_USE=""
export END_EPOCH="1"
export WEIGHT_DECAY="1"

envsubst < $SUBMIT_PATH/infconfig.yaml > $1/inf_config.yaml
}

function build_args_txt {
echo "-c $1/inf_config.yaml">> $SUBMIT_PATH/inf_args.txt
}

for dir in runs/*/ ; do 
    DIR=$(realpath -s $dir)
    populate_template $DIR
    build_args_txt $DIR
    break 1
    #echo "$dir"
done


if [ -n "$1" ]
then
    condor_submit $1 "$SUBMIT_PATH"/inf_submit.sub
else 
    python inference_main.py -c $(realpath -s $DIR)/inf_config.yaml
fi













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
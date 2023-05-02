# Get global path of script. Make every other path relative.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Where to build arg files
SUBMIT_PATH=$SCRIPT_DIR/cluster_utils
if [ -n "$1" ]; then REPEATS=$1; else REPEATS=1; fi

WEIGHT_DECAY_ARRAY=(0.1)

rm -f $SCRIPT_DIR/../tmp_logs/*
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
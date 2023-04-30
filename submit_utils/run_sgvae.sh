UTILS_PATH="submit_utils"
now=`date +"%Y-%m-%d:%S"`
RUN_FOLDER="runs_$now"

rm -f tmp_logs/*
rm -f $UTILS_PATH/sgvae_args.txt

if [ -n "$1" ]; then REPEATS=$1; else REPEATS=1; fi

####  CREATE PARAMETER DICTIONARY   ####
## -Make each parameter a key in the  ##
## dictionary. Set as value or array. ##
DATA_PATH_ARR="/fast/richardson/robot_grasp_data/"
DATASET_ARR="new"
declare -a ACTION_REPETITIONS_ARR=(1 2)
declare -a STYLE_COEF_ARR=(0.1 0.5 1 5 10 15 20)
declare -a KEEP_STYLE_ARR=(True False)
declare -a UPDATE_PRIOR_ARR=(True False)
declare -a BETA_VAE_ARR=(0.1 0.033 0.01 0.0033 0.001 0.00033 0.0001 0.000033 0.00001)
declare -a param_names=(
    "DATA_PATH_ARR" 
    "DATASET_ARR" 
    "ACTION_REPETITIONS_ARR"
    "STYLE_COEF_ARR" 
    "KEEP_STYLE_ARR" 
    "UPDATE_PRIOR_ARR" 
    "BETA_VAE_ARR"
    )
########################################
### variables for tracking folder ID and current run config
COUNTER=0
declare -A RUN_PARAMS

### function to populate the config file
function populate_template {
path=$RUN_FOLDER/$COUNTER
export SAVE_PATH="$path"
for key in ${!RUN_PARAMS[@]}; do
    var_name="${key::-4}"
    var_value="${RUN_PARAMS[${key}]}"
    export $var_name=$var_value

done

mkdir -p $path
envsubst < $UTILS_PATH/sgvae_config.yaml > $path/sgvae_config.yaml
echo "-c $path/sgvae_config.yaml">> $UTILS_PATH/sgvae_args.txt
COUNTER=$[COUNTER+1]
#exit 0
}

function recursive_parameters(){
    local groups=("$@")
    local group=${groups[0]}
    local lst="$group[@]"
    #echo "group name: ${group} with group members: ${!lst}"
    for element in "${!lst}"; do
        #echo -en "\tworking on $element of the $group group\n"    
        RUN_PARAMS[$group]=$element
        if (( ${#groups[@]} > 1 )); then
            recursive_parameters ${groups[@]:1}
        else
            populate_template
        fi
    done
}

recursive_parameters "${param_names[@]}"

if [ -n "$2" ]
then
    condor_submit_bid $2 "$UTILS_PATH"/sgvae_submit.sub
else
    echo "doing nothing"
    #python inference_main.py -c $(realpath -s $DIR)/inf_config.yaml --save_path=run$rep
fi




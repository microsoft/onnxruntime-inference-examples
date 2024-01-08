#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --model_input=*)
          model_input=$(echo $var |cut -f2 -d=)
      ;;
      --model_output=*)
          model_output=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --dataset=*)
          dataset=$(echo $var |cut -f2 -d=)
      ;;
      --algorithm=*)
          algorithm=$(echo $var |cut -f2 -d=)
      ;;
      --block_size=*)
          block_size=$(echo $var |cut -f2 -d=)
      ;;
      --is_symmetric=*)
          is_symmetric=$(echo $var |cut -f2 -d=)
      ;;
      --accuracy_level=*)
          accuracy_level=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    # Check if the model_input ends with the filename extension ".onnx"
    if [[ $model_input =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        model_input=$(dirname "$model_input")
    fi

    # Check if the model_output ends with the filename extension ".onnx"
    if [[ $model_output =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        model_output=$(dirname "$model_output")
    fi

    # Check if the directory exists
    if [ ! -d "$model_output" ]; then
        # If the directory doesn't exist, create it
	mkdir -p "$model_output"
	echo "Created directory $model_output"
    fi

    python main.py \
            --model_input ${model_input} \
            --model_output ${model_output} \
            --batch_size ${batch_size-1} \
            --dataset ${dataset-NeelNanda/pile-10k} \
            --algorithm ${algorithm-RTN} \
            --block_size ${block_size-32} \
            --is_symmetric ${is_symmetric-False} \
            --accuracy_level ${accuracy_level-0} \
            --quantize
}

main "$@"


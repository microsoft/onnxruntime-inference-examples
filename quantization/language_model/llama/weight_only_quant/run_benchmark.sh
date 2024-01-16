#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --model_input=*)
          model_input=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    # Check if the model_input ends with the filename extension ".onnx"
    if [[ $model_input =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        model_input=$(dirname "$model_input")
    fi

    python main.py \
            --model_input ${model_input} \
            --batch_size=${batch_size-1} \
            --tasks=${tasks-lambada_openai} \
            --benchmark
            
}

main "$@"


import argparse
import logging
import numpy
import torch
from pathlib import Path

import data_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="./test_input",
        help="Specify the destination folder of randomly generated input data sets.",
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        choices=range(2, 500),
        default=10,
        help="Specify how many batches of input data sets to generate.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Input batch size")
    parser.add_argument("--past_sequence_length", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=2)

    args = parser.parse_args()
    return args


def main():
    # Process input parameters and setup model input data reader
    args = get_args()

    # Prepare output folder for storing input data files
    output_folder = Path(args.output_dir)
    if not output_folder.exists():
        output_folder.mkdir()
    elif not output_folder.is_dir():
        logging.error(f"File '{str(output_folder)}' exists and is not a folder!")
        return

    # Generate num_batches sets of input data
    num_batches = 1 if args.num_batches < 1 else args.num_batches
    for batch_id in range(num_batches):
        data_file = output_folder / f"batch_{batch_id}.npz"
        if data_file.exists():
            logging.error(
                f"File '{data_file}' exists! Can't write generated input data!"
            )
            return

        input_ids, attention_mask, position_ids, past = data_utils.get_dummy_inputs(
            batch_size=args.batch_size,
            past_sequence_length=args.past_sequence_length,
            sequence_length=args.sequence_length,
            num_attention_heads=16,
            hidden_size=1024,
            num_layer=24,
            vocab_size=50257,
            device="cpu",
            has_position_ids=True,
            has_attention_mask=True,
            input_ids_dtype=torch.int64,
            position_ids_dtype=torch.int64,
            attention_mask_dtype=torch.int64,
        )
        ort_inputs = {
            "input_ids": numpy.ascontiguousarray(input_ids.cpu().numpy()),
            "attention_mask": numpy.ascontiguousarray(attention_mask.cpu().numpy()),
            "position_ids": numpy.ascontiguousarray(position_ids.cpu().numpy()),
        }
        for i, past_i in enumerate(past):
            ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())

        numpy.savez(str(data_file), **ort_inputs)


if __name__ == "__main__":
    main()

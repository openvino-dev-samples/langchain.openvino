from transformers import AutoModel, AutoTokenizer
import openvino as ov
import numpy as np
import torch
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        required=False,
                        default='sentence-transformers/all-mpnet-base-v2',
                        type=str,
                        help='Required. hugging face model id')
    parser.add_argument('-o',
                        '--output',
                        required=False,
                        default="./embedding_model",
                        type=str,
                        help='Required. path to save the ir model')

    args = parser.parse_args()

    model_path = Path(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)

    input_shape = ov.PartialShape([-1, -1])
    dummy_inputs = {"input_ids": torch.ones((1, 10), dtype=torch.long), "attention_mask": torch.ones(
        (1, 10), dtype=torch.long)}
    input_info = [("input_ids", input_shape, np.int64),
                  ("attention_mask", input_shape, np.int64)]

    ov_model = ov.convert_model(model, example_input=dummy_inputs)
    ov.save_model(ov_model, model_path / "openvino_model.xml")

    print(" --- exporting tokenizer --- ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(model_path)

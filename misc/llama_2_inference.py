"""
Demo for Llama-2-7b-hf model from the Hugging Face model hub.
"""

import argparse

import torch
import transformers
from transformers import AutoTokenizer


API_TOKEN = "HERE_COMES_YOUR_HUGGING_FACE_API_TOKEN"

INPUT_LIST = [
    'one, two, three',
    'I was walking home alone, when suddenly',
    'What is a triangle?',
    'Define the term "triangle".',
    'Roses are red, violets are blue',
    'Once upon a time, in a land far far away',
    'What is the capital of France?',
    'Translate the following text to German: "How are you?"',
    '',
]


def main():
    """
    Main function for the demo.
    """

    # build arguments
    args = parse_arguments()

    # load the model and tokenizer (pipeline)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=API_TOKEN)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype=torch.float16,
        device_map=args.device,
        token=API_TOKEN
    )

    # inference model with different inputs
    while True:

        # user input
        input_text = input("Provide input: ")

        # inference model
        output = pipeline(
            input_text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            truncation=True,
            max_length=200
        )

        # logging
        print()
        print(output[0]['generated_text'])
        print()
        print()


def parse_arguments():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Demo for inference a Language Model from the Hugging Face model hub.")

    parser.add_argument("--model_id", default="meta-llama/Llama-2-7b-hf",
                        help="Model ID from the Hugging Face model hub.")

    parser.add_argument("--device", default=torch.device("cuda:7"))

    return parser.parse_args()


if __name__ == "__main__":

    main()

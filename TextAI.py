import os
import subprocess
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '0'


def open_notepad(file_path):
    subprocess.call(['notepad.exe', file_path])

def generate_text(prompt, model_name='gpt2-medium', max_length=400):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors='pt')

    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    input_file_path = 'input_text.txt'
    output_file_path = 'output_text.txt'

    open_notepad(input_file_path)

    with open(input_file_path, 'r') as file:
        user_input = file.read().strip()

    detailed_paragraph = generate_text(user_input)

    with open(output_file_path, 'w') as file:
        file.write(detailed_paragraph)


    open_notepad(output_file_path)

if __name__ == "__main__":
    main()

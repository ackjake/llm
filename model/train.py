from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    pipeline,
    set_seed,
)

# tokenizer = AutoTokenizer.from_pretrained("gemma-2b", local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained("gemma-2b", local_files_only=True)

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))

generator = pipeline('text-generation', model='distilgpt2')
set_seed(48)
print(generator("Why does my peepee come out yellow?", max_length=20, num_return_sequences=3))
set_seed(48)
print(generator("The Black man worked as a", max_length=20, num_return_sequences=3))

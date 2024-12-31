from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer=AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for better performance on GPU
    device_map="auto",  # Automatically choose the device (CPU/GPU)
)


input_text = "What is LangChain?"
inputs=tokenizer(input_text, return_tensors="pt") # Convert text to tokens

# Generate output with the model
outputs=model.generate(inputs['input_ids'], max_length=50, temperature=0.7, top_p=0.95)

# Decode the generated tokens back to text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "moonshotai/Kimi-K2-Instruct", 
    trust_remote_code=True,
    device_map="cuda"
)

save_directory = "./data/kimi-k2-instruct/"  # This will save in the current directory (data folder)
model.save_pretrained(save_directory)
print(f"Model has been saved to {save_directory}")

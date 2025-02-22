# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your selected model and tokenizer
model_name = "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # or specify torch.float16 if desired
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your JSON dataset.
# Ensure that your JSON file ("dpo_train.json") is formatted appropriately (e.g., a JSON list of dictionaries)
train_dataset = load_dataset("json", data_files={"train": "dpo_train.json"}, split="train")

# Define DPO training configuration using DPOConfig
training_args = DPOConfig(
    output_dir="DeepHermes-3-DPO",
    logging_steps=10,
    num_train_epochs=3,               # Adjust as needed
    per_device_train_batch_size=1     # Adjust batch size according to your resources
)

# Initialize the DPOTrainer using the tokenizer as the processing_class
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model to the output directory
trainer.save_model(training_args.output_dir)

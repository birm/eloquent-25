import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load fine-tuned DPO model
model_name = "./DeepHermes-3-DPO"  # Change if saved elsewhere
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Load dataset
df = pd.read_parquet("llama_32_gpt_prompt_distilation.parquet")
df['llm_dpo_quality_pref'] = ''
df['llm_dpo_quality_reasoning'] = ''

# Define prompt template
preamble = """Please tell me which of the two responses (A or B) a human is likely to prefer. Try to make a decisive decision, even if marginal. Please respond with simply A or B."""
rules = """
1. Prioritize factual correctness.
2. Adhere strictly to the prompt's scope.
3. Favor natural conversational tone and clarity.
4. Prefer structured but fluid responses.
"""

# Initialize pipeline for inference
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)

for idx, row in df.iterrows():
    prompt = f"{preamble}{rules}\nInstruction: {row['instruction']}\n\nResponse A: {row['output_a']}\nResponse B: {row['output_b']}\n\nYour choice (A or B):"

    response = pipe(prompt)[0]["generated_text"].strip().upper()

    # Process response
    response = response.replace(".", "").split(" ")[0]  # Normalize formatting
    if response in ['A', 'B']:
        df.at[idx, 'llm_dpo_quality_pref'] = response
    else:
        df.at[idx, 'llm_dpo_quality_pref'] = "UNKNOWN"  # Handle uncertain responses

# Save results
df.to_parquet("llama_32_dpo_evaluation.parquet")

# Compute Agreement Score
df['overall_quality_preference'] = df['overall_quality_preference'].str.upper()
score = sum(df.overall_quality_preference == df.llm_dpo_quality_pref)
print(f"DPO Fine-tuned Model Score: {score}/{len(df)}")

# Identify disagreements
bad_ones = df[
    ((df.overall_quality_preference == "A") & (df.llm_dpo_quality_pref == "B")) |
    ((df.overall_quality_preference == "B") & (df.llm_dpo_quality_pref == "A"))
]

for idx, row in bad_ones.iterrows():
    print(f"The DPO model preferred {row.llm_dpo_quality_pref}, but the human preferred {row.overall_quality_preference}.")

# Optional: Generate a heatmap to compare results
import seaborn as sns
import matplotlib.pyplot as plt

zs_crosstab = pd.crosstab(df['llm_dpo_quality_pref'], df['overall_quality_preference'])
sns.heatmap(zs_crosstab, annot=True, cmap='YlGnBu', fmt='d').set_title('DPO Model vs. Human Preference')
plt.tight_layout()
plt.show()

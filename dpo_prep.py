import pandas as pd
import json

df = pd.read_parquet("llama_32_gpt_prompt_distilation.parquet")

# Format data for DPO training
dpo_data = []
for _, row in df.iterrows():
    dpo_data.append({
        "prompt": row["instruction"],
        "chosen": row["output_a"] if row["overall_quality_preference"] == "A" else row["output_b"],
        "rejected": row["output_b"] if row["overall_quality_preference"] == "A" else row["output_a"]
    })

with open("dpo_train.json", "w") as f:
    json.dump(dpo_data, f, indent=4)

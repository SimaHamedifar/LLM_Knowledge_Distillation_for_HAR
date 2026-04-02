import json
import time
from vllm import LLM, SamplingParams
# import os
# os.environ["VLLM_USE_V1"] = "0"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

def run_teacher_inference(dry_run_count=None):
    """
    Runs the Qwen3-32B as Teacher Agent to generate reasoning and label data. 
    dry_run_count: if not None, runs a dry run test for dry_run_count number of windows. 
    """
    MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
    INPUT_DATA = "preprocessed_data.json"
    PROMPT_TEMPLATE = "LLM_HAR_prompt.txt"

    # INITIALIZE vLLM ENGINE 
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=2,      
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=1500,
        stop=["User:", "<|im_end|>", "\n\n"]
    )

    with open(PROMPT_TEMPLATE, 'r') as f:
        system_prompt = f.read()

    with open(INPUT_DATA, 'r') as f:
        windowed_data = json.load(f)
    
    if dry_run_count is not None:
        print(f"Running in Dry Run mode for {dry_run_count} windows of data.")
        windowed_data = windowed_data[:dry_run_count]
    
    prompts = []
    for window in windowed_data:
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nEvents: {json.dumps(window)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(prompt)
-
    start_time = time.time()
    print(f"Distilling knowledge from {len(prompts)} windows...")
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # SAVE RESULTS
    output_file = "distillation_training_data.jsonl"
    with open(output_file, "w") as f:
        for i, output in enumerate(outputs):
            entry = {
                "instruction": "Explain the activity and provide labels.",
                "input": json.dumps(windowed_data[i]),
                "output": output.outputs[0].text
            }
            f.write(json.dumps(entry) + "\n")
            
    end_time = time.time()
    if dry_run_count is not None:
        print(f"\nTotal time for reasoning: {end_time - start_time:.2f} seconds.")
    
    print(f"\nSuccess! Dataset generated in {output_file}")
    return output_file
    
if __name__ == "__main__":
    run_teacher_inference(dry_run_count=1)

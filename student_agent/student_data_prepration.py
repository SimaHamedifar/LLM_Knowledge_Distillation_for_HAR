import json
from datasets import load_dataset

def prepare_distillation_data(input_file, output_file):
    """
    Step 1: Convert raw teacher JSONL into a formatted training set.
    Step 2: Clean the text to ensure ChatML format for the Student.
    """
    formatted_data = []
    
    print(f"Loading teacher data from {input_file}...")
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # The structure of your prompt for the STUDENT
            # We combine the task instruction and the specific sensor data (input)
            user_message = f"{data['instruction']}\n\nEvents: {data['input']}"
            
            # The target for the student is the Teacher's reasoning + label
            assistant_message = data['output']
            
            # CHATML FORMATTING (Required for Qwen/Llama-3 students)
            full_text = (
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_message}<|im_end|>"
            )
            
            formatted_data.append({"text": full_text})

    # Save as a new JSONL that Hugging Face 'datasets' can load instantly
    with open(output_file, 'w') as out_f:
        for entry in formatted_data:
            out_f.write(json.dumps(entry) + "\n")
            
    print(f"Necessity steps complete. Prepared {len(formatted_data)} samples in {output_file}")

# EXECUTE PREPARATION
if __name__ == "__main__":
    prepare_distillation_data(
        input_file="distillation_training_data.jsonl", 
        output_file="student_ready_data.jsonl"
    )
### Knowledge Distillation-based Human Activity Recognition

The purpose of this project is to use ambient sensor data for human activity recognition leveraging the world knowledge of LLMs.
LLMs knowldge is distilled to smaller language models to increase scalability and efficiency [1]. 

[1] Cumin, Julien, Oussama Er-Rahmany, and Xi Chen. "Knowledge Distillation for LLM-Based Human Activity Recognition in Homes." arXiv preprint arXiv:2601.07469 (2026).

The infromation about the dataset of activities of daily living in a multi-resident setting can be found [here](https://jcumin.github.io/datasets). 

The required Python module are determined in requirements.txt. 

## 1. Data Preparation and Preprocessing
Sensor State Generation and Window Segmentation in preprocess.py

## 2. Teacher Model Inference
The large-scale model Qwen3-32B creates the teacher agent for collecting the reasoning and labels. 
The open-source library, vLLM is used for fast and efficient LLM inference is used. It manages GPU memory more efficiently than HuggingFace Transformers. 
Ubuntu app is used to configure vLLM in Linux environment. 
``` bash
mkdir ~/LLM_Knowledge_Distillation_for_HAR && cd ~/LLM_Knowledge_Distillation_for_HAR
python3 -m venv .venv
source .venv/bin/activate
sudo apt install nvidia-cuda-toolkit build-essential -y
uv pip install vllm
python3 -c "import vllm; print('vLLM Version:', vllm.__version__)"
```
To bridge between Linux and Windows, Windows Subsystem for Linux (WSL2) is applied by installing the related extention on VSCode.   
Check the number of GPUs using 
``` bash
nvidia-smi
``` 
and sanity check of 
```bash
python3 -c "import torch; print(f'GPUs Detected: {torch.cuda.device_count()}');"
```
The autorization for using the Qwen3-32B model is require
``` bash
hf auth login
# The token string is asked.
```
## 3. Knowledge Distillation via LoRA Fine-Tuning
To ensure high-fidelity knowledge transfer, a cross-validation filter was applied to the teacher-generated reasoning traces. Only rationales that successfully converged on the ground-truth activity label were included in the student's fine-tuning corpus.

requirement:
```bash
pip install trl peft
```
## 4. Evaluation and Label Extraction



# LLM Ad Integration Project

This project integrates large language model (LLM) generation with ad relevance scoring, reward modeling, and reinforcement learning. 

---

## Project Structure
```
llm-ad-integration/
├── archived/                  # all the archived files are here
├── data/                      # data is here
├── notebooks/                 # colab/jupyternotebooks
├── scripts/                   # scripts to generate merged data for reponse generation and judge
├── .env.example               # Template for required environment variables
├── README.md                  # Setup instructions
├── requirements.txt           # Python dependencies
├── runpod.yaml                # RunPod config (if applicable)
├── src/
│   ├── main.py                # Main orchestration script
│   ├── generate/              # Response generation
│   ├── judge/                 # Coherence, Helpfulness, Salience, Detectability
│   ├── reward/                # Reward functions (WIP)
│   └── utils/                 # Shared utilities
```

--- 
## 1. Set Up RunPod Container (This is only if we are using RunPod)

1. Go to [runpod.io](https://runpod.io/) and launch a container (e.g., PyTorch template, 16GB or 24GB GPU). (if we don't have yet, CHECK IN WITH TEAM FIRST)

2. On RunPod:
- Enable SSH Access
- Copy the SSH connection command provided (you'll need this later)

3. On your local machine, 
- If you haven't gotten an SSH key, generate one:
  ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
  ```
- Press Enter to accept the default location: '~/.ssh/id_ed25519'
- Add your public key to RunPod: 
  ```bash
    cat ~/.ssh/id_ed25519.pub
  ```
- Copy the output and paste it into RunPod's SSH settings.

- Open your terminal and run:
   ```bash
   ssh -i ~/.ssh/id_ed25519 your_pod_user@ssh.runpod.io
   ```
4. Alternatively, open VSCode or Cursor → `Remote SSH` → use same credentials to connect. (You can click the '><' button on the bottom left of the screen -> Connect to Host - Remote SSH -> + Add New SSH Host -> Copy the 2nd SSH Key in the Runpod -> Paste -> Enter until you are able to clone the git repo)


### Reconnect After Restart
Your pod's IP may change after restarting.
Update your local `~/.ssh/config` like so:
```ssh
Host runpod
  HostName NEW_IP_HERE
  User root
  Port (maybe have this, look at the IP in the connect tab)
  IdentityFile ~/.ssh/id_ed25519
```
Maybe we can automate this with a Python or shell script.

---

## 2. Set Up Virtual Environment - Both Locally and Remotely (On RunPod)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4. Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Fill in the required keys:
```ini
OPENAI_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_if_needed
```

---

## 5. Run the Pipeline

```bash
python src/main.py
```
This script will:
- Generate responses (with/without ads) using a base LLM.
- Have AI judges (Gemini 1.5 Flash, OpenAI for embeddings) evaluate these responses.
- Use Reinforcement Learning (PPO) to fine-tune the ad-insertion model based on AI feedback.
- Log detailed metrics and save checkpoints for resuming training.

---

## Codebase Overview

This project aims to train a language model to seamlessly and effectively insert advertisements into conversational responses. It uses d-RLAIF (Reinforcement Learning from AI Feedback), where another AI (an LLM Judge) provides the feedback signal to guide the training.

Key components include:

*   **Ad Generation (`src/generate/`):**
    *   `generator.py`: Uses a base LLM (e.g., Llama 3.1-8B) to generate responses with and without ads, guided by structured prompts.
    *   `prompts.py`: Contains the detailed instructions and rules for the ad-insertion model.
*   **LLM Judging (`src/judge/`):**
    *   An LLM judge (Gemini 1.5 Flash) evaluates ad-inserted responses based on:
        *   `coherence.py`: How well the ad fits the conversation.
        *   `helpfulness.py`: If the overall response remains helpful.
        *   `salience.py`: Clarity and noticeability of the ad.
        *   `detectability.py`: How obviously it's an ad (uses OpenAI embeddings for similarity).
    *   `utils.py`: Manages API calls to judges, embedding generation, and caching.
*   **PPO Training Loop (`src/training/manual_ppo_loop.py`):**
    *   The core RL process using Proximal Policy Optimization (PPO).
    *   `DataProcessor`: Manages data flow, generation, judging, reward calculation, and model updates.
    *   **Batched Training Updates:** To stabilize training, losses from individual queries are accumulated across a training batch (defined by `TRAINING_BATCH_SIZE` in `src/config.py`). A single optimizer step is then performed using the aggregated loss for the entire batch, rather than updating the model after each query.
    *   Extensive logging of generations, scores, and metrics to the `logs/` directory.
*   **Checkpoint Management (`src/training/checkpoint_manager.py`):**
    *   **Comprehensive State Saving:** Full checkpoints, which include the model's state dictionary, the tokenizer's state, and the optimizer's state dictionary, are saved periodically (controlled by `CHECKPOINT_INTERVAL_BATCHES` in `src/config.py`) *after a set number of training batches have been completed*. This ensures that the saved model state reflects learning from complete batches.
    *   **Granular Resumption:** Training can be seamlessly resumed. The system first looks for the latest valid full model checkpoint. Crucially, a `last_query_position.json` file within the `CHECKPOINT_DIR` (e.g., `checkpoints/ppo_llama/`) is updated after *every processed query*. This allows the training to restart from the exact point it left off by re-processing any queries from an interrupted batch, even if the interruption occurred before a full model checkpoint was due.
    *   **Metrics and Best Model Tracking:** A `training_metrics.json` file in the `CHECKPOINT_DIR` logs validation scores and other metrics over time. It also keeps track of the checkpoint that achieved the best validation reward, allowing for easy identification of the top-performing model state.
    *   **Automated Cleanup:** To manage disk space, the system automatically retains only a configurable number (`CHECKPOINTS_TO_KEEP` in `src/config.py`) of the most recent full checkpoints, deleting older ones.
    *   All checkpoint-related files and directories are managed within the path specified by `CHECKPOINT_DIR` in `src/config.py`.
*   **Configuration (`src/config.py`):**
    *   Centralized settings for API keys, base model name, checkpoint directory, and data file paths.
*   **Entry Point (`src/main.py`):**
    *   Orchestrates setup and starts the PPO training loop.
    *   Includes `sys.path` modifications for direct script execution.

---

## API Call Mechanisms

The system interacts with several external APIs:

*   **LLM Judges (Google Gemini 1.5 Flash):**
    *   Managed in `src/judge/utils.py` using the `google-generativeai` library.
    *   `GOOGLE_API_KEY` is loaded from `.env`.
    *   Features `async_parallel_judge_responses` using `asyncio.gather` for concurrent API calls to all four judge aspects (coherence, helpfulness, salience, detectability), significantly speeding up evaluation.
*   **OpenAI Embeddings (for Detectability Judge):**
    *   Managed in `src/judge/utils.py` using the `openai` library.
    *   `OPENAI_API_KEY` is loaded from `.env`.
    *   `get_embedding` function calls the OpenAI embeddings API (e.g., "text-embedding-ada-002").
*   **Hugging Face Hub (Model/Tokenizer Downloads):**
    *   Handled by the `transformers` library (`AutoModelForCausalLM.from_pretrained()`, etc.) in various modules.
    *   `HF_TOKEN` from `.env` is used, crucial for accessing gated models.

---

## Potential Enhancements & Future Work

While the current system is robust, several areas could be explored for further improvement:

*   **Refined Reward Function:**
    *   Experiment with weighted averages for judge scores.
    *   Introduce specific rewards/penalties for desirable/undesirable behaviors.
    *   Implement reward shaping (dynamic changes to the reward function).
*   **Advanced PPO Implementation:**
    *   Integrate a Value Network and Generalized Advantage Estimation (GAE).
    *   Implement the full PPO clipped surrogate objective and an entropy bonus for exploration.
*   **"No Ad" Decision Intelligence:**
    *   Train the model or judges to determine when *not* to insert an ad.
*   **Human Feedback Integration:**
    *   Explore Direct Preference Optimization (DPO) using human preference data.
    *   Fine-tune AI judges based on human ratings of their scores.
*   **Dynamic Ad Selection:**
    *   Implement a pre-step to select the most relevant ad from a larger database.
*   **Automated & Diverse Evaluation Suite:**
    *   Develop a fixed, diverse test set for consistent generalization measurement.
    *   Include "challenge" queries to test edge cases.
*   **Cost-Performance Optimization for Judging:**
    *   Experiment with smaller/faster judge models or more complex single-judge prompts.
*   **User Context Awareness:**
    *   Incorporate conversation history for more contextually relevant ad insertions.

---

## 6. Cleanup (Only for RunPod)

To save GPU space:
```bash
huggingface-cli cache purge
```

---

## Notes
- Always store secrets in `.env` and **never push them to GitHub**.
- `.venv`, `__pycache__`, and `.env` should be excluded via `.gitignore`.
- Do not hardcode RunPod IPs — update your SSH config instead.

------
## ATTENTION 
To properly stop the training script and ensure final checkpoints and logs are saved, please press `Ctrl+C` in the terminal where the script is running. The script is designed to catch this interruption and perform a graceful shutdown.

If you have to stop running, please manually save the following critical folders to ensure you can resume your progress:
- The entire directory specified by `CHECKPOINT_DIR` in `src/config.py` (e.g., `checkpoints/ppo_llama` if using the default setting). This directory contains all model checkpoints, optimizer states, and the crucial `last_query_position.json` for exact resumption.
- The `logs/` directory, which contains all generation, judging, training, and statistics CSV files.

When you start a new runtime or switch environments and want to resume:
1. Ensure your `src/config.py` has the correct `CHECKPOINT_DIR` pointing to where your *intended* checkpoint data will reside.
2. If you are restoring from a backup, completely replace the existing `CHECKPOINT_DIR` in your workspace with the version you saved.
3. Similarly, replace the `logs/` directory with your saved version if you want to preserve historical log data.

This ensures that the `CheckpointManager` and `DataProcessor` can find all necessary files to resume training and logging correctly.



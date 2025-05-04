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
- Copy the output and paste it into RunPod’s SSH settings.

- Open your terminal and run:
   ```bash
   ssh -i ~/.ssh/id_ed25519 your_pod_user@ssh.runpod.io
   ```
4. Alternatively, open VSCode or Cursor → `Remote SSH` → use same credentials to connect.

### Reconnect After Restart
Your pod’s IP may change after restarting.
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
- Generate responses (with/without ads)
- Judge coherence/helpfulness/salience
- Prepare for reward modeling

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

+-----------------------------------------------------+
|        Step 1: Load Data (merged_queries_ads.csv)   |
+-------------------------------+---------------------+
                                |
                                v
+-----------------------------------------------------+
|  Step 2: generate_responses(query, ad_facts, model) |
|        → Response Without Ad (A)                    |
|        → Response With Ad (B)                       |
+-----------------------------------------------------+
                                |
                                v
+-----------------------------------------------------+
|  Step 3: Judging                                     |
|   ├─ judge_detectability(A, B)                       |
|   ├─ judge_coherence(B, query)                       |
|   ├─ judge_helpfulness(query, B)                     |
|   └─ judge_salience(query, B, ad_facts)              |
+-----------------------------------------------------+
                                |
                                v
+-----------------------------------------------------+
| Step 4: compute_reward()                             |
|   = Sum of judge scores + (1 - detectability)        |
+-----------------------------------------------------+
                                |
                                v
+-----------------------------------------------------+
| Step 5: PPOTrainer.step(query_tensor, B, reward)     |
|   (Fine-tune policy model)                           |
+-----------------------------------------------------+
                                |
                                v
+-----------------------------------------------------+
| 🔁 Step 5.5: Resample Past Query (every N steps)     |
|   • Pick a past query                                |
|   • Generate response with *updated* model           |
|   • Compute new reward & log                         |
+-----------------------------------------------------+
                                |
                                v
+-----------------------------------------------------+
| Step 6: Logging & Model Saving                       |
|   • Save reward + breakdown to CSV                   |
|   • Periodically save model + tokenizer              |
+-----------------------------------------------------+
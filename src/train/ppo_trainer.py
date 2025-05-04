from transformers import AutoTokenizer
import torch
from judge.coherence import judge_coherence
from judge.helpfulness import judge_helpfulness
from judge.salience import judge_ad_salience
from judge.detectability import judge_detectability
import random
import csv
from datetime import datetime
import os

class PPOTrainer:
    def __init__(self, model, tokenizer_name="your-model-name", save_every=100):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.save_every = save_every
        
        # Create directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        # Initialize logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/ppo_training_log_{timestamp}.csv"
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'query', 'response', 'coherence', 'helpfulness', 
                           'salience', 'detectability', 'total_reward'])

    def save_checkpoint(self, step):
        """Save model and tokenizer checkpoints"""
        checkpoint_dir = f"checkpoints/step_{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        print(f"Saved checkpoint at step {step} to {checkpoint_dir}")

    def compute_reward(self, query, response, ad_facts):
        # Run all judges
        coherence = judge_coherence(response, query)
        helpfulness = judge_helpfulness(query, response)
        salience = judge_ad_salience(query, response, ad_facts)
        
        # For detectability, we need both with and without ad responses
        # Since we don't have the without-ad response here, we'll skip it
        # You may want to modify this based on your needs
        detectability = 0.5  # Placeholder
        
        # Extract scores
        coherence_score = coherence.get("Coherence Score", 0)
        helpfulness_score = helpfulness.get("H1", 0)
        salience_score = salience.get("Ad Salience Score", 0)
        
        # Compute total reward
        reward = (
            coherence_score +
            helpfulness_score +
            salience_score +
            (1 - detectability)  # Lower detectability is better
        )
        
        return reward, {
            "coherence": coherence_score,
            "helpfulness": helpfulness_score,
            "salience": salience_score,
            "detectability": detectability
        }

    def step(self, query_tensor, response_tensor, reward):
        # Your PPO update logic here
        # This is a placeholder - implement your actual PPO update
        pass

    def train(self, training_data, num_steps, resample_interval=100):
        past_queries = []
        
        for step, (query, ad_facts) in enumerate(training_data):
            # Step 1: Generate response
            input_ids = self.tokenizer(query, return_tensors="pt").input_ids
            response_tensor = self.model.generate(input_ids, max_new_tokens=100)
            response = self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)
            
            # Step 2: Compute reward
            reward, scores = self.compute_reward(query, response, ad_facts)
            
            # Step 3: PPO update
            query_tensor = self.tokenizer(query, return_tensors="pt").input_ids
            self.step(query_tensor, response_tensor, torch.tensor([reward]))
            
            # Step 4: Log results
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, query, response,
                    scores["coherence"],
                    scores["helpfulness"],
                    scores["salience"],
                    scores["detectability"],
                    reward
                ])
            
            # Step 5: Save checkpoint periodically
            if step % self.save_every == 0 and step > 0:
                self.save_checkpoint(step)
            
            # Store query for resampling
            past_queries.append(query)
            
            # Step 6: Resample every N steps
            if step % resample_interval == 0 and step > 0:
                past_query = random.choice(past_queries)
                past_ids = self.tokenizer(past_query, return_tensors="pt").input_ids
                new_response_tensor = self.model.generate(past_ids, max_new_tokens=100)
                new_response = self.tokenizer.decode(new_response_tensor[0], skip_special_tokens=True)
                
                # Compute new reward
                new_reward, new_scores = self.compute_reward(past_query, new_response, ad_facts)
                
                # Log resampled results
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"resample_{step}", past_query, new_response,
                        new_scores["coherence"],
                        new_scores["helpfulness"],
                        new_scores["salience"],
                        new_scores["detectability"],
                        new_reward
                    ]) 
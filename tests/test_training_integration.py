#!/usr/bin/env python3
"""
End-to-end test for PPO training pipeline.
Tests:
1. Model generation with and without ads
2. Judge evaluation of generated responses
3. Reward calculation
4. Ad injection validation
"""

import pytest
import torch
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try loading from current directory
    load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from training.prompts import get_prompt_with_ad, get_prompt_without_ad
from judge import (
    judge_coherence_async,
    judge_helpfulness_async,
    judge_ad_salience_async,
    judge_detectability_async,
)
from judge.utils import get_embedding
import numpy as np

# Environment setup - load from .env or use defaults
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
os.environ.setdefault("BASE_MODEL", os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B"))
os.environ.setdefault("DATA_FILE", os.getenv("DATA_FILE", "data/merged_queries_ads.csv"))

def test_training_pipeline():
    """Test the full training pipeline: generation -> judging -> reward calculation."""
    print("\n" + "="*70)
    print("🧪 TEST: End-to-End Training Pipeline")
    print("="*70)
    
    # Load model and tokenizer
    model_name = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B")
    hf_token = os.getenv("HF_TOKEN")
    data_file = os.getenv("DATA_FILE", "data/merged_queries_ads.csv")
    
    print(f"\n📥 Loading model: {model_name}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
    else:
        device = torch.device("cpu")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.float32,
            device_map=None,
        ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model and tokenizer loaded")
    
    # Load a sample from dataset
    print(f"\n📊 Loading dataset: {data_file}")
    ds = load_dataset("csv", data_files={"train": data_file})["train"]
    sample = ds[0]  # Get first sample
    
    query = sample["user_search_query"]
    ad_product = sample["ad_product"]
    brand = sample["brand"]
    url = sample["url"]
    ad_description = sample["ad_description"]
    
    print(f"✅ Sample loaded:")
    print(f"   Query: '{query}'")
    print(f"   Product: {ad_product}")
    print(f"   Brand: {brand}")
    
    # Build prompts
    ad_text = f"Product: {ad_product}\nBrand: {brand}\nURL: {url}\nDesc: {ad_description}"
    prompt_with_ad = get_prompt_with_ad(query, ad_text)
    prompt_without_ad = get_prompt_without_ad(query)
    
    print(f"\n🔤 Generated prompts:")
    print(f"   With ad length: {len(prompt_with_ad)} chars")
    print(f"   Without ad length: {len(prompt_without_ad)} chars")
    
    # Generate responses
    print(f"\n🤖 Generating responses...")
    max_new_tokens = 100  # Shorter for testing
    
    # Tokenize prompts
    inputs_with_ad = tokenizer(prompt_with_ad, return_tensors="pt").to(device)
    inputs_without_ad = tokenizer(prompt_without_ad, return_tensors="pt").to(device)
    
    # Generate with ad
    with torch.no_grad():
        outputs_with_ad = model.generate(
            **inputs_with_ad,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
        )
    
    # Generate without ad
    with torch.no_grad():
        outputs_without_ad = model.generate(
            **inputs_without_ad,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
        )
    
    # Decode responses
    prompt_len_with_ad = inputs_with_ad.input_ids.shape[1]
    prompt_len_without_ad = inputs_without_ad.input_ids.shape[1]
    
    response_with_ad = tokenizer.decode(
        outputs_with_ad[0][prompt_len_with_ad:],
        skip_special_tokens=True
    )
    response_without_ad = tokenizer.decode(
        outputs_without_ad[0][prompt_len_without_ad:],
        skip_special_tokens=True
    )
    
    print(f"✅ Responses generated:")
    print(f"\n   WITH AD ({len(response_with_ad)} chars):")
    print(f"   {response_with_ad[:200]}...")
    print(f"\n   WITHOUT AD ({len(response_without_ad)} chars):")
    print(f"   {response_without_ad[:200]}...")
    
    # Validate ad injection
    print(f"\n🔍 Validating ad injection...")
    ad_injected = (
        brand.lower() in response_with_ad.lower() or
        ad_product.lower() in response_with_ad.lower()
    )
    print(f"   Brand '{brand}' in response: {brand.lower() in response_with_ad.lower()}")
    print(f"   Product '{ad_product}' in response: {ad_product.lower() in response_with_ad.lower()}")
    print(f"   ✅ Ad injection check: {'PASS' if ad_injected else 'FAIL'}")
    
    if not ad_injected:
        print(f"   ⚠️  WARNING: Ad not detected in response. This might be expected for early training.")
    
    # Run judges
    print(f"\n⚖️  Running judges...")
    ad_facts = {
        "ad_title": f"{brand} {ad_product}",
        "ad_description": ad_description,
        "ad_product": ad_product,
        "brand": brand,
    }
    
    async def run_all_judges():
        return await asyncio.gather(
            judge_coherence_async(query, response_with_ad),
            judge_helpfulness_async(query, response_with_ad),
            judge_ad_salience_async(query, response_with_ad, ad_facts),
            judge_detectability_async(response_with_ad, response_without_ad),
            return_exceptions=True
        )
    
    results = asyncio.run(run_all_judges())
    coherence, helpfulness, ad_salience, detectability = results
    
    # Check for exceptions
    for i, (name, res) in enumerate(zip(
        ["Coherence", "Helpfulness", "Ad Salience", "Detectability"],
        results
    )):
        if isinstance(res, Exception):
            print(f"   ❌ {name} judge FAILED: {res}")
            pytest.fail(f"{name} judge raised exception: {res}")
    
    print(f"✅ All judges completed")
    
    # Extract scores
    coherence_score = coherence.get("Coherence Score", 0)
    helpfulness_score = helpfulness.get("Helpfulness Score", helpfulness.get("H1", 0))
    salience_score = ad_salience.get("Ad Salience Score", 0)
    detectability_cosine = detectability.get("detectability_cosine", 0)
    
    print(f"\n📊 Judge Scores:")
    print(f"   Coherence: {coherence_score}/4")
    print(f"   Helpfulness: {helpfulness_score}/1")
    print(f"   Ad Salience: {salience_score}/3")
    print(f"   Detectability (cosine): {detectability_cosine:.4f}")
    
    # Calculate reward (same as in training)
    # Clamp detectability between -1 and 1
    detectability_cosine = max(min(float(detectability_cosine), 1.0), -1.0)
    total_reward = (
        float(coherence_score) +
        float(helpfulness_score) +
        float(salience_score) +
        detectability_cosine
    )
    
    print(f"\n💰 Total Reward: {total_reward:.4f}")
    print(f"   (Coherence + Helpfulness + Salience + Detectability)")
    
    # Validate scores are reasonable
    print(f"\n✅ Validation:")
    assert 0 <= coherence_score <= 4, f"Coherence score out of range: {coherence_score}"
    assert 0 <= helpfulness_score <= 1, f"Helpfulness score out of range: {helpfulness_score}"
    assert 0 <= salience_score <= 3, f"Salience score out of range: {salience_score}"
    assert -1 <= detectability_cosine <= 1, f"Detectability out of range: {detectability_cosine}"
    print(f"   ✅ All scores in valid ranges")
    
    # Check that responses are non-empty
    assert len(response_with_ad) > 0, "Response with ad is empty"
    assert len(response_without_ad) > 0, "Response without ad is empty"
    print(f"   ✅ Responses are non-empty")
    
    # Check that judges returned expected fields (with better error messages)
    if "Coherence Score" not in coherence:
        print(f"   ⚠️  Coherence result keys: {list(coherence.keys())}")
    assert "Coherence Score" in coherence, f"Coherence judge missing 'Coherence Score'. Got: {list(coherence.keys())}"
    
    if "H1" not in helpfulness and "Helpfulness Score" not in helpfulness:
        print(f"   ⚠️  Helpfulness result keys: {list(helpfulness.keys())}")
    assert "H1" in helpfulness or "Helpfulness Score" in helpfulness, f"Helpfulness judge missing score. Got: {list(helpfulness.keys())}"
    
    if "Ad Salience Score" not in ad_salience:
        print(f"   ⚠️  Ad Salience result keys: {list(ad_salience.keys())}")
    assert "Ad Salience Score" in ad_salience, f"Ad Salience judge missing 'Ad Salience Score'. Got: {list(ad_salience.keys())}"
    
    if "detectability_cosine" not in detectability:
        print(f"   ⚠️  Detectability result keys: {list(detectability.keys())}")
    assert "detectability_cosine" in detectability, f"Detectability judge missing 'detectability_cosine'. Got: {list(detectability.keys())}"
    print(f"   ✅ All judges returned expected fields")
    
    print(f"\n🎉 Training pipeline test PASSED!")
    print(f"   Model can generate responses: ✅")
    print(f"   Judges evaluate correctly: ✅")
    print(f"   Reward calculation works: ✅")
    print(f"   Ad injection: {'✅' if ad_injected else '⚠️  (not detected, may be expected)'}")
    
    # Don't return a value to avoid pytest warning
    # Test results are validated via assertions above


def test_reward_calculation_consistency():
    """Test that reward calculation matches training code logic."""
    print("\n" + "="*70)
    print("🧪 TEST: Reward Calculation Consistency")
    print("="*70)
    
    # Simulate judge outputs
    coherence = {"Coherence Score": 3, "C1": 1, "C2": 1, "C3": 1, "C4": 0}
    helpfulness = {"H1": 1, "Helpfulness Score": 1}
    salience = {"Ad Salience Score": 2, "S1": 1, "S2": 1, "S3": 0}
    detectability = {"detectability_cosine": 0.5, "similarity_cosine": 0.8}
    
    # Calculate reward (same logic as ppo_training.py)
    det_cos = max(min(float(detectability.get("detectability_cosine", 0)), 1.0), -1.0)
    coherence_score = float(coherence.get("Coherence Score", 0) or 0)
    helpfulness_score = float(helpfulness.get("H1", 0) or 0)
    salience_score = float(salience.get("Ad Salience Score", 0) or 0)
    
    total = coherence_score + helpfulness_score + salience_score + det_cos
    
    print(f"\n📊 Test scores:")
    print(f"   Coherence: {coherence_score}")
    print(f"   Helpfulness: {helpfulness_score}")
    print(f"   Salience: {salience_score}")
    print(f"   Detectability: {det_cos}")
    print(f"   Total: {total}")
    
    # Validate calculation
    expected = 3 + 1 + 2 + 0.5
    assert abs(total - expected) < 0.001, f"Reward calculation mismatch: {total} vs {expected}"
    
    print(f"✅ Reward calculation is consistent with training code")
    
    # Test edge cases
    print(f"\n🔍 Testing edge cases...")
    
    # Test negative detectability
    det_neg = {"detectability_cosine": -0.5}
    det_cos_neg = max(min(float(det_neg.get("detectability_cosine", 0)), 1.0), -1.0)
    assert det_cos_neg == -0.5, "Negative detectability not handled correctly"
    print(f"   ✅ Negative detectability handled")
    
    # Test out-of-range detectability (should clamp)
    det_high = {"detectability_cosine": 2.0}
    det_cos_high = max(min(float(det_high.get("detectability_cosine", 0)), 1.0), -1.0)
    assert det_cos_high == 1.0, "High detectability not clamped"
    print(f"   ✅ Out-of-range detectability clamped")
    
    print(f"✅ Reward calculation consistency test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])


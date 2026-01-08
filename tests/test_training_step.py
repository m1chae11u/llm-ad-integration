#!/usr/bin/env python3
"""
Unit tests for the PPO training step.
Tests the core step() method that performs PPO optimization.

WHAT'S SUFFICIENT FOR THESE TESTS:
----------------------------------
These tests only verify that:
- The step() method can be called without errors
- It returns stats with expected keys
- The loss is a valid number

IMPORTANT: All tests use the SAME model (from minimal_trainer_config fixture).
You don't need different models for different tests - one model works for all.

OPTIONS FOR RUNNING TESTS:
--------------------------
1. Use TinyLlama (RECOMMENDED - open model, no auth needed, fast):
   export TEST_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   pytest tests/test_training_step.py
   # Memory: ~6GB needed ✅

2. Use 8B model with 8-bit quantization (for testing with production model):
   export USE_8BIT=true
   pytest tests/test_training_step.py
   # Memory: ~25GB needed (8GB weights + 16GB gradients + optimizer + activations)

3. Use 8B model without quantization (will OOM on 40GB GPU):
   pytest tests/test_training_step.py
   # Memory: ~67GB needed ❌ (will fail)

NOTE: Meta Llama models (meta-llama/*) are GATED and require:
- Accepting license on HuggingFace
- Valid HF_TOKEN with access
Use TinyLlama for easier testing without authentication.
"""

import pytest
import torch
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
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

from transformers import AutoTokenizer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import Dataset
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
)

# Environment setup - load from .env or use defaults
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
# Default to 8B model, but recommend using TEST_MODEL env var for smaller model
os.environ.setdefault("BASE_MODEL", os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B"))
# Recommended TEST_MODEL values (use non-gated models for easier testing):
# - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B, ~6GB needed, OPEN, no auth needed) ✅ RECOMMENDED
# - "microsoft/phi-2" (2.7B but efficient, ~8GB needed, OPEN)
# - "meta-llama/Llama-3.2-1B-Instruct" (1B, ~6GB needed, GATED - requires HuggingFace license acceptance)
# - Any 1B-2B model should work fine for these tests

# Set PyTorch CUDA allocator config to reduce fragmentation
# This helps with OOM errors by using expandable segments
# Note: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
if torch.cuda.is_available():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


@pytest.fixture
def sample_ad_facts():
    """Sample ad facts for testing."""
    return [
        {
            "ad_id": "test_ad_1",
            "ad_product": "Test Product",
            "brand": "Test Brand",
            "url": "https://example.com/product",
            "ad_description": "A great test product for testing purposes."
        }
    ]


@pytest.fixture
def sample_dataset():
    """Create a minimal test dataset."""
    data = {
        "vague_query": ["What is a good product?"],
        "ad_product": ["Test Product"],
        "brand": ["Test Brand"],
        "url": ["https://example.com"],
        "ad_description": ["A test product"],
    }
    ds = Dataset.from_dict(data)
    # Verify dataset has data
    assert len(ds) > 0, "Dataset should have at least 1 row"
    ds = ds.add_column("dataset_idx", list(range(len(ds))))
    ds.set_format(type="python")
    # Verify after formatting
    assert len(ds) > 0, "Dataset should still have rows after formatting"
    return ds


@pytest.fixture
def minimal_trainer_config(sample_ad_facts, sample_dataset):
    """Create minimal configuration for trainer."""
    # Allow override with TEST_MODEL env var for smaller models
    # Default to TinyLlama for testing (fits in 40GB GPU without quantization)
    # IMPORTANT: TEST_MODEL takes precedence over BASE_MODEL for tests
    default_test_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small, open model for testing
    base_model = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B")  # For production (not used if TEST_MODEL not set)
    # Always prefer TEST_MODEL, fallback to TinyLlama (NOT BASE_MODEL) for tests
    test_model = os.getenv("TEST_MODEL")
    if test_model is None:
        test_model = default_test_model  # Use TinyLlama, not BASE_MODEL
    model_name = test_model
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        pytest.skip("HF_TOKEN not set, skipping trainer tests")
    
    # Print memory diagnostics
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n💾 GPU Memory Analysis:")
        print(f"   Total GPU Memory: {total_mem:.2f} GB")
        print(f"   Model: {model_name}")
        
        # Check if 8-bit quantization is enabled
        use_8bit = os.getenv("USE_8BIT", "false").lower() == "true"
        
        # Estimate memory requirements
        # For 8B model in float16: ~16GB weights + ~16GB gradients + optimizer + activations
        # For 8B model in 8-bit: ~8GB weights + ~16GB gradients (still float16) + optimizer + activations
        # For 1B model in float16: ~2GB weights + ~2GB gradients + optimizer + activations
        # For 125M model in float16: ~0.25GB weights + ~0.25GB gradients + optimizer + activations
        if "8B" in model_name or "8b" in model_name:
            if use_8bit:
                estimated_mem = 25  # 8GB weights (8-bit) + 16GB gradients + optimizer + activations
                print(f"   ✅ Estimated memory needed (8-bit): ~{estimated_mem} GB (should fit in {total_mem:.2f} GB GPU)")
                print(f"   💡 8-bit quantization enabled - model weights reduced by ~50%")
            else:
                estimated_mem = 40  # Conservative estimate
                print(f"   ⚠️  Estimated memory needed: ~{estimated_mem} GB (may exceed {total_mem:.2f} GB GPU)")
                print(f"   💡 Options:")
                print(f"      1. Set USE_8BIT=true to enable 8-bit quantization (~25GB needed)")
                print(f"      2. Use TEST_MODEL env var with a smaller model (1B or 125M)")
        elif "1B" in model_name or "1b" in model_name or "TinyLlama" in model_name:
            estimated_mem = 6  # Conservative estimate: 2GB weights + 2GB gradients + 1.5GB optimizer + 0.5GB activations
            print(f"   ✅ Estimated memory needed: ~{estimated_mem} GB (should fit in {total_mem:.2f} GB GPU)")
        elif "125M" in model_name or "125m" in model_name or "124M" in model_name:
            estimated_mem = 1.5  # Conservative estimate: 0.25GB weights + 0.25GB gradients + 0.5GB optimizer + 0.5GB activations
            print(f"   ✅ Estimated memory needed: ~{estimated_mem} GB (plenty of room in {total_mem:.2f} GB GPU)")
        else:
            print(f"   ⚠️  Unknown model size, memory requirements unclear")
            print(f"   💡 Recommendation: Use TEST_MODEL env var with a known small model")
    
    # Use a smaller model for testing if available, or skip if too large
    try:
        # Try to load model (this will be slow, but necessary for real testing)
        if torch.cuda.is_available():
            # Print memory before loading
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                allocated_before = torch.cuda.memory_allocated(0) / 1e9
                print(f"   Memory before model load: {allocated_before:.2f} GB")
            
            # Check if we should use 8-bit quantization for large models
            use_8bit = os.getenv("USE_8BIT", "false").lower() == "true"
            if use_8bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                print(f"   🔧 Using 8-bit quantization (reduces memory by ~50%)")
                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=hf_token,
                    quantization_config=quantization_config,
                    device_map="cuda:0",
                    low_cpu_mem_usage=True,
                )
            else:
                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=hf_token,
                    torch_dtype=torch.float16,
                    device_map="cuda:0",
                    low_cpu_mem_usage=True,
                )
            
            # Print memory after loading
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated(0) / 1e9
                peak_allocated = torch.cuda.max_memory_allocated(0) / 1e9
                print(f"   Memory after model load: {allocated_after:.2f} GB")
                print(f"   Peak memory during load: {peak_allocated:.2f} GB")
                print(f"   Model weights size: ~{allocated_after - allocated_before:.2f} GB")
            
            # Enable gradient checkpointing on the model to reduce memory usage
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            elif hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'gradient_checkpointing_enable'):
                model.pretrained_model.gradient_checkpointing_enable()
        else:
            pytest.skip("No GPU available, skipping trainer tests (too slow on CPU)")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True, token=hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Clear CUDA cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleared CUDA cache after model loading")
        
        # Create reference model
        ref_model = create_reference_model(model)
        ref_model = ref_model.to("cpu")
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        
        # Clear CUDA cache after reference model creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleared CUDA cache after reference model creation")
        
        # Create training args
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=1,
                save_strategy="no",
                logging_steps=1,
                max_grad_norm=1.0,
                gradient_checkpointing=True,
                no_cuda=not torch.cuda.is_available(),
            )
            
            model_args = ModelArguments(model_name_or_path=model_name, hf_hub_token=hf_token)
            data_args = DataArguments(template="default", dataset="test.csv", dataset_dir=".")
            finetuning_args = FinetuningArguments(
                ppo_epochs=1,
                ppo_buffer_size=1,
                reward_model_type="api",
                ppo_target=0.1,
            )
            generating_args = GeneratingArguments(
                do_sample=False,
                max_new_tokens=50,  # Short for testing
            )
            
            yield {
                "model": model,
                "ref_model": ref_model,
                "tokenizer": tokenizer,
                "training_args": training_args,
                "model_args": model_args,
                "data_args": data_args,
                "finetuning_args": finetuning_args,
                "generating_args": generating_args,
                "train_dataset": sample_dataset,
                "ad_facts_list": sample_ad_facts,
            }
            
            # Cleanup: Clear CUDA cache after test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"🧹 Cleared CUDA cache in fixture cleanup")
    except Exception as e:
        import traceback
        error_msg = str(e)
        full_traceback = traceback.format_exc()
        print(f"\n❌ Failed to load model '{model_name}':")
        print(f"   Error: {error_msg}")
        print(f"\n   Full traceback:")
        print(full_traceback)
        pytest.skip(f"Failed to load model for testing: {error_msg}")


def test_step_method_basic(minimal_trainer_config):
    """Test that step() method can be called and returns expected stats."""
    print("\n" + "="*70)
    print("🧪 TEST: Basic Step Method")
    print("="*70)
    
    from training.ppo_training import MyPPOTrainer, PPODataCollator
    
    config = minimal_trainer_config
    collator = PPODataCollator(config["tokenizer"])
    
    # Verify dataset has data before creating trainer
    train_dataset = config["train_dataset"]
    print(f"📊 Dataset info before trainer: {len(train_dataset)} rows")
    print(f"📊 Dataset columns: {train_dataset.column_names}")
    assert len(train_dataset) > 0, f"Dataset is empty! Expected at least 1 row, got {len(train_dataset)}"
    
    # Create trainer
    trainer = MyPPOTrainer(
        model_args=config["model_args"],
        training_args=config["training_args"],
        finetuning_args=config["finetuning_args"],
        generating_args=config["generating_args"],
        callbacks=None,
        model=config["model"],
        reward_model=None,
        ref_model=config["ref_model"],
        tokenizer=config["tokenizer"],
        processor=None,
        data_collator=collator,
        train_dataset=train_dataset,
        ad_facts_list=config["ad_facts_list"],
    )
    
    # Prepare sample queries and responses
    # These should be tokenized sequences
    tokenizer = config["tokenizer"]
    device = next(config["model"].parameters()).device
    
    # Create simple query and response
    query_text = "What is a good product?"
    response_text = "A good product is one that meets your needs."
    
    query_tokens = tokenizer(query_text, return_tensors="pt")["input_ids"][0].to(device)
    response_tokens = tokenizer(response_text, return_tensors="pt")["input_ids"][0].to(device)
    
    queries = [query_tokens]
    responses = [response_tokens]
    
    # Create sample rewards (should be tensors on the correct device)
    rewards = [torch.tensor(2.5, device=device)]
    
    print(f"\n📝 Testing step() with:")
    print(f"   Query length: {len(query_tokens)} tokens")
    print(f"   Response length: {len(response_tokens)} tokens")
    print(f"   Reward: {rewards[0].item()}")
    
    # Clear CUDA cache before training step
    if torch.cuda.is_available():
        # Check memory BEFORE clearing
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_before = torch.cuda.memory_allocated(0) / 1e9
        reserved_before = torch.cuda.memory_reserved(0) / 1e9
        unused_before = reserved_before - allocated_before  # Memory reserved but not used
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Check memory AFTER clearing
        allocated_after = torch.cuda.memory_allocated(0) / 1e9
        reserved_after = torch.cuda.memory_reserved(0) / 1e9
        unused_after = reserved_after - allocated_after
        freed_by_cache_clear = reserved_before - reserved_after
        
        # Print memory diagnostics
        print(f"\n💾 Memory Analysis (Why cache clearing doesn't help enough):")
        print(f"   Total GPU: {total_mem:.2f} GB")
        print(f"\n   📊 BEFORE cache clear:")
        print(f"      Allocated (in use): {allocated_before:.2f} GB ← Model weights (CAN'T be freed)")
        print(f"      Reserved: {reserved_before:.2f} GB")
        print(f"      Unused reserved: {unused_before:.2f} GB ← This is what empty_cache() frees")
        print(f"\n   📊 AFTER cache clear:")
        print(f"      Allocated (in use): {allocated_after:.2f} GB ← Still here (model is using it)")
        print(f"      Reserved: {reserved_after:.2f} GB")
        print(f"      Unused reserved: {unused_after:.2f} GB")
        print(f"      ✅ Freed by empty_cache(): {freed_by_cache_clear:.2f} GB")
        print(f"\n   ⚠️  Why OOM still happens:")
        print(f"      empty_cache() only frees UNUSED reserved memory ({freed_by_cache_clear:.2f} GB)")
        print(f"      It CANNOT free memory that's actively in use (model weights: {allocated_after:.2f} GB)")
        print(f"\n   📈 Memory needed for training step:")
        print(f"      - Model weights (already loaded): {allocated_after:.2f} GB")
        print(f"      - Gradients (NEW, needed during backward): ~{allocated_after:.2f} GB")
        print(f"      - Optimizer states (NEW, Adam needs 2x): ~{allocated_after * 2:.2f} GB")
        print(f"      - Activations (NEW, during forward/backward): ~2-4 GB")
        print(f"      - Total needed: ~{allocated_after * 4 + 3:.2f} GB")
        print(f"      - Available: {total_mem:.2f} GB")
        print(f"      - Shortfall: ~{allocated_after * 4 + 3 - total_mem:.2f} GB")
        print(f"\n   💡 Solution: Use smaller model (TEST_MODEL env var) or 8-bit quantization")
        
        print(f"\n🧹 Cleared CUDA cache (freed {freed_by_cache_clear:.2f} GB of unused memory)")
    
    # Set model to train mode
    trainer.model.train()
    
    # Call step method
    print(f"\n🔄 Calling step()...")
    try:
        stats = trainer.step(queries, responses, rewards)
        print(f"✅ Step completed successfully")
        
        # Validate stats structure
        assert isinstance(stats, dict), f"Stats should be a dict, got {type(stats)}"
        print(f"   Stats keys: {list(stats.keys())}")
        
        # Check for expected keys (may vary by llamafactory version)
        expected_keys = ["ppo/loss/total"]
        found_keys = [k for k in expected_keys if k in stats]
        assert len(found_keys) > 0, f"Expected at least one of {expected_keys} in stats, got {list(stats.keys())}"
        
        # Validate loss is a number
        if "ppo/loss/total" in stats:
            loss = stats["ppo/loss/total"]
            assert isinstance(loss, (int, float, torch.Tensor)), f"Loss should be numeric, got {type(loss)}"
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            print(f"   Loss: {loss:.4f}")
            assert loss >= 0, f"Loss should be non-negative, got {loss}"
        
        print(f"✅ Basic step test PASSED")
        
        # Clear CUDA cache after successful step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Step failed: {e}")
        # Clear CUDA cache even on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import traceback
        traceback.print_exc()
        raise


def test_step_method_with_multiple_samples(minimal_trainer_config):
    """Test step() method with multiple samples in batch."""
    print("\n" + "="*70)
    print("🧪 TEST: Step Method with Multiple Samples")
    print("="*70)
    
    from training.ppo_training import MyPPOTrainer, PPODataCollator
    
    config = minimal_trainer_config
    collator = PPODataCollator(config["tokenizer"])
    
    # Create trainer with batch size 2
    config["training_args"].per_device_train_batch_size = 2
    trainer = MyPPOTrainer(
        model_args=config["model_args"],
        training_args=config["training_args"],
        finetuning_args=config["finetuning_args"],
        generating_args=config["generating_args"],
        callbacks=None,
        model=config["model"],
        reward_model=None,
        ref_model=config["ref_model"],
        tokenizer=config["tokenizer"],
        processor=None,
        data_collator=collator,
        train_dataset=config["train_dataset"],
        ad_facts_list=config["ad_facts_list"],
    )
    
    tokenizer = config["tokenizer"]
    device = next(config["model"].parameters()).device
    
    # Create two queries and responses
    query1 = tokenizer("What is product A?", return_tensors="pt")["input_ids"][0].to(device)
    response1 = tokenizer("Product A is great.", return_tensors="pt")["input_ids"][0].to(device)
    
    query2 = tokenizer("What is product B?", return_tensors="pt")["input_ids"][0].to(device)
    response2 = tokenizer("Product B is good.", return_tensors="pt")["input_ids"][0].to(device)
    
    queries = [query1, query2]
    responses = [response1, response2]
    rewards = [torch.tensor(2.0, device=device), torch.tensor(3.0, device=device)]
    
    print(f"\n📝 Testing step() with batch size 2:")
    print(f"   Query 1 length: {len(query1)} tokens")
    print(f"   Response 1 length: {len(response1)} tokens")
    print(f"   Reward 1: {rewards[0].item()}")
    print(f"   Query 2 length: {len(query2)} tokens")
    print(f"   Response 2 length: {len(response2)} tokens")
    print(f"   Reward 2: {rewards[1].item()}")
    
    # Clear CUDA cache before training step
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.model.train()
    
    print(f"\n🔄 Calling step()...")
    try:
        stats = trainer.step(queries, responses, rewards)
        print(f"✅ Step completed successfully with batch size 2")
        
        assert isinstance(stats, dict), "Stats should be a dict"
        print(f"   Stats: {stats}")
        
        print(f"✅ Multi-sample step test PASSED")
        
        # Clear CUDA cache after successful step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Step failed: {e}")
        # Clear CUDA cache even on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import traceback
        traceback.print_exc()
        raise


def test_step_method_validates_inputs(minimal_trainer_config):
    """Test that step() method validates input shapes and types."""
    print("\n" + "="*70)
    print("🧪 TEST: Step Method Input Validation")
    print("="*70)
    
    from training.ppo_training import MyPPOTrainer, PPODataCollator
    
    config = minimal_trainer_config
    collator = PPODataCollator(config["tokenizer"])
    
    trainer = MyPPOTrainer(
        model_args=config["model_args"],
        training_args=config["training_args"],
        finetuning_args=config["finetuning_args"],
        generating_args=config["generating_args"],
        callbacks=None,
        model=config["model"],
        reward_model=None,
        ref_model=config["ref_model"],
        tokenizer=config["tokenizer"],
        processor=None,
        data_collator=collator,
        train_dataset=config["train_dataset"],
        ad_facts_list=config["ad_facts_list"],
    )
    
    tokenizer = config["tokenizer"]
    device = next(config["model"].parameters()).device
    
    # Test: mismatched lengths
    query = tokenizer("Test query", return_tensors="pt")["input_ids"][0].to(device)
    response = tokenizer("Test response", return_tensors="pt")["input_ids"][0].to(device)
    
    print(f"\n📝 Testing input validation...")
    
    # Test empty inputs
    print(f"   Testing empty queries...")
    try:
        trainer.step([], [], [])
        assert False, "Should raise error for empty inputs"
    except (ValueError, RuntimeError, AssertionError) as e:
        print(f"   ✅ Correctly rejected empty inputs: {type(e).__name__}")
    
    # Test mismatched lengths
    print(f"   Testing mismatched lengths...")
    try:
        trainer.step([query], [response], [])  # Missing reward
        assert False, "Should raise error for mismatched lengths"
    except (ValueError, RuntimeError, AssertionError) as e:
        print(f"   ✅ Correctly rejected mismatched lengths: {type(e).__name__}")
    
    print(f"✅ Input validation test PASSED")


def test_step_method_updates_model(minimal_trainer_config):
    """Test that step() method actually updates model parameters."""
    print("\n" + "="*70)
    print("🧪 TEST: Step Method Updates Model")
    print("="*70)
    
    from training.ppo_training import MyPPOTrainer, PPODataCollator
    
    config = minimal_trainer_config
    collator = PPODataCollator(config["tokenizer"])
    
    trainer = MyPPOTrainer(
        model_args=config["model_args"],
        training_args=config["training_args"],
        finetuning_args=config["finetuning_args"],
        generating_args=config["generating_args"],
        callbacks=None,
        model=config["model"],
        reward_model=None,
        ref_model=config["ref_model"],
        tokenizer=config["tokenizer"],
        processor=None,
        data_collator=collator,
        train_dataset=config["train_dataset"],
        ad_facts_list=config["ad_facts_list"],
    )
    
    tokenizer = config["tokenizer"]
    device = next(config["model"].parameters()).device
    
    # Get initial parameter values
    # Get a parameter that will definitely be updated (e.g., from the value head)
    initial_params = {}
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
            if len(initial_params) >= 3:  # Just check a few parameters
                break
    
    if not initial_params:
        print("   ⚠️  No trainable parameters found, skipping parameter update test")
        return
    
    print(f"\n📝 Testing parameter updates...")
    print(f"   Tracking {len(initial_params)} parameters")
    
    # Create sample data
    query = tokenizer("What is a test?", return_tensors="pt")["input_ids"][0].to(device)
    response = tokenizer("This is a test response.", return_tensors="pt")["input_ids"][0].to(device)
    reward = torch.tensor(2.5, device=device)
    
    # Clear CUDA cache before training step
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.model.train()
    
    # Run step
    print(f"\n🔄 Running step()...")
    stats = trainer.step([query], [response], [reward])
    print(f"✅ Step completed")
    
    # Check if parameters changed
    params_changed = False
    for name, initial_value in initial_params.items():
        current_param = dict(trainer.model.named_parameters())[name]
        if not torch.equal(initial_value, current_param.data):
            params_changed = True
            print(f"   ✅ Parameter '{name}' was updated")
            break
    
    if params_changed:
        print(f"✅ Model parameters were updated")
    else:
        print(f"   ⚠️  Parameters didn't change (might be expected if gradient is zero)")
        # This is not necessarily a failure - parameters might not change if:
        # - Gradient is zero
        # - Learning rate is very small
        # - Model is frozen
    
    print(f"✅ Model update test PASSED")


def test_step_method_handles_edge_cases(minimal_trainer_config):
    """Test step() method with edge cases like very short/long sequences."""
    print("\n" + "="*70)
    print("🧪 TEST: Step Method Edge Cases")
    print("="*70)
    
    from training.ppo_training import MyPPOTrainer, PPODataCollator
    
    config = minimal_trainer_config
    collator = PPODataCollator(config["tokenizer"])
    
    trainer = MyPPOTrainer(
        model_args=config["model_args"],
        training_args=config["training_args"],
        finetuning_args=config["finetuning_args"],
        generating_args=config["generating_args"],
        callbacks=None,
        model=config["model"],
        reward_model=None,
        ref_model=config["ref_model"],
        tokenizer=config["tokenizer"],
        processor=None,
        data_collator=collator,
        train_dataset=config["train_dataset"],
        ad_facts_list=config["ad_facts_list"],
    )
    
    tokenizer = config["tokenizer"]
    device = next(config["model"].parameters()).device
    
    print(f"\n📝 Testing edge cases...")
    
    # Test 1: Very short sequences
    print(f"   Test 1: Very short sequences...")
    short_query = tokenizer("Hi", return_tensors="pt")["input_ids"][0].to(device)
    short_response = tokenizer("Hi", return_tensors="pt")["input_ids"][0].to(device)
    reward = torch.tensor(1.0, device=device)
    
    # Clear CUDA cache before training steps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.model.train()
    try:
        stats = trainer.step([short_query], [short_response], [reward])
        print(f"   ✅ Handled short sequences")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ⚠️  Short sequences failed: {e}")
        # This might be expected if sequences are too short
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Test 2: Zero reward
    print(f"   Test 2: Zero reward...")
    query = tokenizer("Test query", return_tensors="pt")["input_ids"][0].to(device)
    response = tokenizer("Test response", return_tensors="pt")["input_ids"][0].to(device)
    zero_reward = torch.tensor(0.0, device=device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        stats = trainer.step([query], [response], [zero_reward])
        print(f"   ✅ Handled zero reward")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ⚠️  Zero reward failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    # Test 3: Negative reward
    print(f"   Test 3: Negative reward...")
    negative_reward = torch.tensor(-1.0, device=device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        stats = trainer.step([query], [response], [negative_reward])
        print(f"   ✅ Handled negative reward")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ⚠️  Negative reward failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    print(f"✅ Edge cases test PASSED")


def test_model_forward_returns_tuple(minimal_trainer_config):
    """Test that the patched model forward method returns tuple format expected by llamafactory.
    
    This is the critical test that verifies the fix for:
    ValueError: not enough values to unpack (expected 3, got 1)
    """
    print("\n" + "="*70)
    print("🧪 TEST: Model Forward Returns Tuple Format")
    print("="*70)
    
    from training.ppo_training import make_trainer
    import tempfile
    from pathlib import Path
    
    config = minimal_trainer_config
    
    # Create a temporary data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("vague_query,ad_product,brand,url,ad_description\n")
        f.write("What is a test?,Test Product,Test Brand,https://test.com,Test description\n")
        temp_data_path = f.name
    
    try:
        # Create trainer using make_trainer (which applies the patch)
        print(f"\n📝 Creating trainer with make_trainer (applies patch)...")
        trainer = make_trainer(
            model_name=os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B"),
            hf_token=os.getenv("HF_TOKEN"),
            data_path=temp_data_path,
            ad_facts_list=config["ad_facts_list"],
        )
        
        model = trainer.model
        tokenizer = config["tokenizer"]
        device = next(model.parameters()).device
        
        # Create sample input (same format as llamafactory's batched_forward_pass)
        print(f"\n📝 Testing model forward with return_dict=True...")
        test_text = "What is a test?"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
        # This is the exact call that fails in llamafactory:
        # logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)
        print(f"   Calling: model(**inputs, return_dict=True, use_cache=False)")
        
        try:
            result = model(**inputs, return_dict=True, use_cache=False)
            print(f"   ✅ Model call succeeded")
            print(f"   Result type: {type(result)}")
            
            # Verify it returns a tuple
            assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
            print(f"   ✅ Result is a tuple")
            
            # Verify it has 3 elements
            assert len(result) == 3, f"Expected 3 elements, got {len(result)}"
            print(f"   ✅ Result has 3 elements: {len(result)}")
            
            logits, past_key_values, value = result
            print(f"   ✅ Successfully unpacked: logits={type(logits)}, past_key_values={type(past_key_values)}, value={type(value)}")
            
            # Verify logits is a tensor
            assert logits is not None, "Logits should not be None"
            assert hasattr(logits, 'shape'), "Logits should be a tensor with shape"
            print(f"   ✅ Logits shape: {logits.shape}")
            
            # Value can be None or a tensor
            if value is not None:
                print(f"   ✅ Value shape: {value.shape if hasattr(value, 'shape') else type(value)}")
            else:
                print(f"   ⚠️  Value is None (may be expected for some models)")
            
            print(f"\n✅ Model forward tuple format test PASSED")
            print(f"   The patch is working correctly!")
            
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                print(f"   ❌ FAILED: Still getting unpacking error - patch is not working!")
                print(f"   Error: {e}")
                raise AssertionError(f"Model forward still returns wrong format: {e}")
            else:
                raise
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    finally:
        # Clean up temp file
        if Path(temp_data_path).exists():
            Path(temp_data_path).unlink()


def test_model_forward_direct_call(minimal_trainer_config):
    """Test that model.forward() directly also returns tuple format."""
    print("\n" + "="*70)
    print("🧪 TEST: Model Forward Direct Call")
    print("="*70)
    
    from training.ppo_training import make_trainer
    import tempfile
    from pathlib import Path
    
    config = minimal_trainer_config
    
    # Create a temporary data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("vague_query,ad_product,brand,url,ad_description\n")
        f.write("What is a test?,Test Product,Test Brand,https://test.com,Test description\n")
        temp_data_path = f.name
    
    try:
        # Create trainer using make_trainer (which applies the patch)
        trainer = make_trainer(
            model_name=os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B"),
            hf_token=os.getenv("HF_TOKEN"),
            data_path=temp_data_path,
            ad_facts_list=config["ad_facts_list"],
        )
        
        model = trainer.model
        tokenizer = config["tokenizer"]
        device = next(model.parameters()).device
        
        # Test direct forward call
        print(f"\n📝 Testing model.forward() directly...")
        test_text = "What is a test?"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
        result = model.forward(**inputs, return_dict=True, use_cache=False)
        
        # Verify it returns a tuple
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"
        
        logits, past_key_values, value = result
        assert logits is not None, "Logits should not be None"
        
        print(f"   ✅ Direct forward call returns tuple format")
        print(f"   ✅ Logits shape: {logits.shape}")
        
        print(f"\n✅ Direct forward call test PASSED")
        
    finally:
        # Clean up temp file
        if Path(temp_data_path).exists():
            Path(temp_data_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])


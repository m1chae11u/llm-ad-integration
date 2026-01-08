#!/usr/bin/env python3
"""
Test script to validate all critical components before full training.
Run this before starting PPO training to catch issues early.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

def test_embeddings():
    """Test Google embedding API."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Google Embeddings API")
    print("="*60)
    
    try:
        from judge.utils import get_embedding
        
        test_text = "This is a test sentence for embedding generation."
        print(f"📝 Testing with text: '{test_text}'")
        
        start_time = time.time()
        embedding = get_embedding(test_text)
        elapsed = time.time() - start_time
        
        print(f"✅ Embedding generated in {elapsed:.2f}s")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.4f}")
        print(f"   Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")
        
        # Validate shape
        if embedding.shape != (1536,):
            print(f"⚠️  WARNING: Expected shape (1536,), got {embedding.shape}")
            return False
        
        # Validate norm (should be ~1.0 if normalized)
        if abs(np.linalg.norm(embedding) - 1.0) > 0.1:
            print(f"⚠️  WARNING: Embedding not normalized (norm={np.linalg.norm(embedding):.4f})")
        
        print("✅ Embeddings test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemini_api():
    """Test Gemini API for judging."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Gemini API (Judge Models)")
    print("="*60)
    
    try:
        from judge.utils import call_gemini_api
        import asyncio
        
        # Test prompt similar to what judges use
        test_prompt = """
        Rate the following response on a scale of 1-10 for helpfulness.
        Response: "This product is great for cooking."
        Return only a JSON object with a 'score' field.
        """
        
        print(f"📝 Testing Gemini API with judge prompt...")
        
        start_time = time.time()
        result = asyncio.run(call_gemini_api(test_prompt, ["score"], max_retries=2))
        elapsed = time.time() - start_time
        
        print(f"✅ Gemini API call completed in {elapsed:.2f}s")
        print(f"   Result: {result}")
        
        # Check for score in various possible formats
        has_score = (
            result.get("score") is not None or
            result.get("Score") is not None or
            result.get("SCORE") is not None
        )
        if not has_score:
            print(f"⚠️  WARNING: No score returned in result: {result}")
            print("   This might be okay if the model returns data in different format")
        
        print("✅ Gemini API test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model loading and device detection."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Model Loading & Device Detection")
    print("="*60)
    
    try:
        import torch
        import os
        from trl import AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer
        
        model_name = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B")
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            print("⚠️  SKIPPED: HF_TOKEN not set")
            return None
        
        print(f"📝 Loading model: {model_name}")
        
        if torch.cuda.is_available():
            print(f"   GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            try:
                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_auth_token=hf_token,
                    torch_dtype=torch.float16,
                    device_map="cuda:0",
                )
                
                # Check device
                if hasattr(model, 'pretrained_model'):
                    base_model = model.pretrained_model
                    if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
                        device = base_model.model.embed_tokens.weight.device
                    else:
                        device = next(base_model.parameters()).device
                else:
                    device = next(model.parameters()).device
                
                print(f"✅ Model loaded successfully")
                print(f"   Device: {device}")
                
                if device.type == "cpu":
                    print(f"❌ ERROR: Model is on CPU but GPU is available!")
                    return False
                elif device.type == "cuda":
                    print(f"✅ Model is on GPU: {device}")
                    return True
                else:
                    print(f"⚠️  Unknown device type: {device}")
                    return False
                    
            except Exception as e:
                print(f"❌ Model loading FAILED: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("⚠️  SKIPPED: No GPU available")
            return None
            
    except Exception as e:
        print(f"❌ Model loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_speed():
    """Test generation speed."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Generation Speed Test")
    print("="*60)
    
    try:
        import torch
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        
        model_name = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B")
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            print("⚠️  SKIPPED: HF_TOKEN not set")
            return None
        
        if not torch.cuda.is_available():
            print("⚠️  SKIPPED: No GPU available")
            return None
        
        print(f"📝 Loading model for generation test...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=hf_token,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
            use_auth_token=hf_token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test prompt
        test_prompt = "What is the meaning of life?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
        
        print(f"📝 Generating with prompt: '{test_prompt}'")
        print(f"   Max new tokens: 50 (quick test)")
        
        generation_config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Warmup
        print("   Warming up...")
        with torch.no_grad():
            _ = model.generate(**inputs, generation_config=generation_config, max_new_tokens=10)
        torch.cuda.synchronize()
        
        # Actual test
        print("   Running generation test...")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        
        print(f"✅ Generation completed")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Tokens generated: {num_tokens}")
        print(f"   Speed: {num_tokens/elapsed:.2f} tokens/sec")
        print(f"   Generated text: {generated_text[:100]}...")
        
        # Check if speed is reasonable
        if elapsed > 30:
            print(f"⚠️  WARNING: Generation is slow ({elapsed:.2f}s). Expected < 30s for 50 tokens.")
            print(f"   This suggests model might be on CPU or GPU not being used properly.")
            return False
        
        if num_tokens/elapsed < 5:
            print(f"⚠️  WARNING: Generation speed is very slow ({num_tokens/elapsed:.2f} tokens/sec)")
            print(f"   Expected > 10 tokens/sec on GPU")
            return False
        
        print("✅ Generation speed test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Generation speed test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading."""
    print("\n" + "="*60)
    print("🧪 TEST 5: Data Loading")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        data_path = "data/merged_queries_ads.csv"
        
        if not Path(data_path).exists():
            print(f"⚠️  SKIPPED: Data file not found at {data_path}")
            return None
        
        print(f"📝 Loading dataset from {data_path}...")
        
        ds = load_dataset("csv", data_files={"train": data_path})["train"]
        ds = ds.add_column("dataset_idx", list(range(len(ds))))
        ds.set_format(type="python")
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Number of samples: {len(ds)}")
        print(f"   Columns: {list(ds.column_names)}")
        
        # Check a sample
        sample = ds[0]
        print(f"   Sample keys: {list(sample.keys())}")
        
        required_keys = ["user_search_query", "ad_title", "ad_description"]
        missing = [k for k in required_keys if k not in sample]
        if missing:
            print(f"❌ ERROR: Missing required keys: {missing}")
            return False
        
        print("✅ Data loading test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🚀 COMPONENT TEST SUITE")
    print("="*60)
    print("Testing all critical components before PPO training...")
    
    results = {}
    
    # Run tests
    results['embeddings'] = test_embeddings()
    results['gemini_api'] = test_gemini_api()
    results['model_loading'] = test_model_loading()
    results['generation_speed'] = test_generation_speed()
    results['data_loading'] = test_data_loading()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}: FAILED")
            failed += 1
        else:
            print(f"⚠️  {test_name}: SKIPPED")
            skipped += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All critical tests passed! Ready for training.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


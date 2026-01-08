# Test Suite

Test files to validate all critical components before running full PPO training.

## Quick Start

Run all component tests:
```bash
python3 tests/test_components.py
```

Test judge integration:
```bash
python3 tests/test_judge_integration.py
```

Test training step:
```bash
python3 tests/test_training_step.py
```

## Test Files

### `test_components.py`
Comprehensive test suite that validates:
1. **Embeddings API** - Tests Google embedding generation
2. **Gemini API** - Tests judge model API calls
3. **Model Loading** - Tests model loading and GPU device detection
4. **Generation Speed** - Tests actual generation speed on GPU
5. **Data Loading** - Tests dataset loading and format

### `test_judge_integration.py`
Tests all 4 judges working together:
- Coherence judge
- Helpfulness judge
- Ad Salience judge
- Detectability judge

### `test_training_step.py`
Unit tests for the PPO training step (`step()` method):
- **Basic step test** - Validates that `step()` can be called and returns expected stats
- **Multi-sample test** - Tests step with batch size > 1
- **Input validation** - Tests that step validates input shapes and types
- **Model updates** - Verifies that step actually updates model parameters
- **Edge cases** - Tests with very short sequences, zero/negative rewards

These tests validate the core PPO optimization step without running the full training loop.

**Note:** Requires GPU and HF_TOKEN. Tests will be skipped if these are not available.

### `test_training_integration.py`
End-to-end integration test for the full training pipeline:
- Model generation with and without ads
- Judge evaluation of generated responses
- Reward calculation
- Ad injection validation

This tests the complete flow from data → generation → judging → reward calculation.

## Expected Results

### ✅ All Tests Pass
If all tests pass, you're ready for training:
```
🎉 All critical tests passed! Ready for training.
```

### ⚠️ Warnings
Some tests may show warnings but still pass. Common warnings:
- Embedding normalization (should be ~1.0)
- Missing score fields (API might return different format)

### ❌ Failures
If tests fail, check:
1. **Environment variables**: `HF_TOKEN`, `GOOGLE_API_KEY` set correctly
2. **GPU availability**: `nvidia-smi` shows GPU
3. **API keys**: Valid and have proper permissions
4. **Data file**: `data/merged_queries_ads.csv` exists

## Troubleshooting

### Embeddings Fail
- Check `GOOGLE_API_KEY` is set
- Try installing: `pip install google-genai`
- Check API quota/limits

### Gemini API Fails
- Check `GOOGLE_API_KEY` is set
- Verify model names are available in your region
- Check API quota/limits

### Model Loading Fails
- Check `HF_TOKEN` is set
- Verify model name is correct
- Check GPU memory is free

### Generation Too Slow
- Verify model is on GPU (check logs)
- Check GPU utilization with `nvidia-smi`
- May need to reduce batch size

## Running Before Training

**Always run these tests before starting full training!**

They catch issues early and save hours of debugging.


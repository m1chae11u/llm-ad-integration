#!/bin/bash
# Quick script to run all tests

echo "🧪 Running Component Tests..."
echo ""

python3 tests/test_components.py

echo ""
echo "🧪 Running Judge Integration Tests..."
echo ""

python3 tests/test_judge_integration.py

echo ""
echo "🧪 Running Training Integration Tests..."
echo ""

python3 tests/test_training_integration.py

echo ""
echo "✅ All tests completed!"


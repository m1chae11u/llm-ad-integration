#!/bin/bash
# Script to clear all GPU memory and reset for a fresh start

echo "🧹 Clearing GPU memory..."

# Kill any Python processes (optional, may require sudo for some processes)
pkill -9 -f python 2>/dev/null || true
sleep 1

# Clear CUDA cache via Python
source venv/bin/activate
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    # Clear all CUDA caches
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.synchronize()
    
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    
    print(f"✅ CUDA cache cleared")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")
else:
    print("⚠️  No CUDA available")
EOF

# Show GPU status
echo ""
echo "📊 GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | \
    awk '{printf "   Used: %d MB / Free: %d MB / Total: %d MB (%.1f%% free)\n", $1, $2, $3, ($2/$3)*100}'

echo ""
echo "✅ Memory cleared! Ready for fresh start."


---

## Critical Fix: Training Exits Immediately Issue

### Problem
Training exits immediately with message: `Saved last checkpoint at update 380000` without performing any actual training, even with `--epochs 10` or higher.

### Root Cause
The pretrained checkpoint file (`model_380000.pt`) contains internal metadata that stores the training state at update 380000. When the script loads this checkpoint, it thinks training is already complete and exits immediately.

### Solution: Reset the Checkpoint

**Step 1: Create a reset script**

Save this as `reset_checkpoint.py` in your `ckpts` folder:
```python
import torch

# Load the checkpoint
checkpoint = torch.load('model_pretrained.pt', map_location='cpu')

# Reset the training state
if 'ema_model_state_dict' in checkpoint:
    # Keep only the model weights, remove training state
    new_checkpoint = {
        'ema_model_state_dict': checkpoint['ema_model_state_dict']
    }
else:
    new_checkpoint = checkpoint

# Save the reset checkpoint
torch.save(new_checkpoint, 'model_pretrained_reset.pt')
print("Checkpoint reset successfully!")
```

**Step 2: Rename your pretrained model**
```cmd
cd C:\Users\User\Desktop\F5-TTS\ckpts
ren model_380000.pt model_pretrained.pt
```

**Step 3: Run the reset script**
```cmd
python reset_checkpoint.py
```

**Step 4: Delete any existing cached checkpoints**
```cmd
cd C:\Users\User\Desktop\F5-TTS\ckpts
rmdir /S /Q my_dataset_ar
```

**Step 5: Use the reset checkpoint for training**
```cmd
cd C:\Users\User\Desktop\F5-TTS
set PYTHONPATH=C:\Users\User\Desktop\F5-TTS\src && ^
accelerate launch --mixed_precision=fp16 ^
  "C:\Users\User\Desktop\F5-TTS\src\f5_tts\train\finetune_cli.py" ^
  --exp_name F5TTS_Base ^
  --learning_rate 1e-4 ^
  --batch_size_type frame --batch_size_per_gpu 1600 ^
  --grad_accumulation_steps 2 --max_samples 0 ^
  --epochs 10 --num_warmup_updates 100 ^
  --save_per_updates 200 --keep_last_n_checkpoints -1 --last_per_updates 100 ^
  --dataset_name my_dataset_ar --finetune ^
  --pretrain "C:\Users\User\Desktop\F5-TTS\ckpts\model_pretrained_reset.pt" ^
  --tokenizer char ^
  --tokenizer_path "C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar_char\vocab.txt" ^
  --log_samples
```

### What the Reset Script Does

The script strips all training metadata (update counter, optimizer state, learning rate schedule) from the checkpoint while preserving the actual neural network weights. This allows fine-tuning to start from update 0 while still leveraging the pretrained model's knowledge.

### Expected Training Output

After applying the fix, you should see:
```
Epoch 1/10: 11% | 50/452 [33:12<4:08:02, 37.02s/update, loss=0.401, update=50]
```

**Key metrics to monitor:**
- **Update counter**: Should start from 0 and increment
- **Loss**: Should be displayed and generally decrease over time
- **Progress percentage**: Should show actual progress through epochs
- **Estimated time**: Should show realistic completion time

### Understanding Training Checkpoints Location

The training script creates checkpoints in:
```
C:\Users\User\Desktop\F5-TTS\ckpts\{dataset_name}\
├─ pretrained_{model_name}.pt  (copy of your pretrained model)
└─ model_last.pt                (saved at each checkpoint interval)
```

If you re-run training, the script will auto-resume from `model_last.pt`. To start fresh, delete the `{dataset_name}` folder.

---

## Understanding Training Parameters

### Key Concepts

**Updates vs Batches:**
- With `--grad_accumulation_steps 2`:
  - 1 update = 2 batches processed + 1 weight adjustment
  - 1000 updates = 2000 batches processed + 1000 weight updates

**Checkpoint numbering:**
- `model_380000.pt` = Model saved after 380,000 training updates (NOT hours)
- This represents several days of continuous training depending on GPU speed

**Frame-based batching:**
- For 24kHz audio with hop=256: frames ≈ seconds × 93.75
- For 12-second clips: ~1,125 frames per sample
- `--batch_size_per_gpu 1600` with 2 accumulation steps = effective batch of 3,200 frames

### Common Warnings (Safe to Ignore)
```
UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec...
```

**This is NOT an error!** It's a deprecation warning about future PyTorch versions. Your training will work perfectly. Focus on the actual training metrics instead.

---

## Summary of Conversation

### Issues Encountered:
1. ✅ Training exiting immediately with "Saved last checkpoint at update 380000"
2. ✅ Checkpoint auto-resume behavior preventing fresh training
3. ✅ Understanding training parameters and metrics

### Solutions Applied:
1. Created checkpoint reset script to strip training metadata
2. Renamed pretrained model to remove update number from filename
3. Deleted cached checkpoint folders that caused auto-resume
4. Used reset checkpoint with cleaned training state

### Final Result:
Training successfully started from update 0 with proper progress tracking and loss calculation.
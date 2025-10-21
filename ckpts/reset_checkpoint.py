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
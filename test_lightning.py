import torch
from src.training.lightning_module import YOLOv8nLightningModule

# Create lightning module
model = YOLOv8nLightningModule()
print('✅ Lightning module created successfully')

# Test that the model can be created and has the right structure
print(f'   Model type: {type(model.model)}')
print(f'   Loss function type: {type(model.loss_fn)}')
print(f'   Learning rate: {model.learning_rate}')

# Test that we can access the model parameters
print(f'   Model parameters: {sum(p.numel() for p in model.parameters()):,}')

print('✅ Lightning module basic functionality verified')

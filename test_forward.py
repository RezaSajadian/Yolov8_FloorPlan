import torch
from src.models.yolo_model import YOLOv8n

# Create model
model = YOLOv8n(num_classes=3, input_size=640)
print('✅ Model created successfully')

# Test forward pass
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    outputs = model(dummy_input)
print('✅ Forward pass successful')
print(f'   Output is a list of {len(outputs)} tensors')
for i, output in enumerate(outputs):
    print(f'   Output {i} shape: {output.shape}')

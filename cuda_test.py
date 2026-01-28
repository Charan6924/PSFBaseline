import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Allocated Memory: {torch.cuda.memory_allocated(0)} bytes")
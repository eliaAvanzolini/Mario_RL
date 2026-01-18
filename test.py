import torch
print(f"CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
exit()
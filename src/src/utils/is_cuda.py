import torch

if torch.cuda.is_available():
    print("cuda is available.")
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}: {torch.cuda.get_device_name(i)}")
else:
    print("cuda not available, using cpu")

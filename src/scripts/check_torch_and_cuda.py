import torch

print("torch path : ", torch.__file__)
print("torch version : ", torch.__version__)
print("nvidia cuda version : ", torch.version.cuda)

cuda_status = torch.cuda.is_available()
print("is cuda available? : ", cuda_status)

if cuda_status:
    print("number of gpus:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}: {torch.cuda.get_device_name(i)}")
else:
    print("cuda not available, using cpu")

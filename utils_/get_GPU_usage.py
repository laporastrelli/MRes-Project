import torch

fs = []
for id in range(4):
    t = torch.cuda.get_device_properties(id).total_memory
    r = torch.cuda.memory_reserved(id)
    a = torch.cuda.memory_allocated(id)
    f = r-a  # free inside reserved
    fs.append(f)

    if f > 0:
        print('CUDA:{id} is available with free memory of {free} bytes.'\
             .format(id=id, free=f))

if sum(fs) == 0:
    print('No available CUDA device at the moment.')
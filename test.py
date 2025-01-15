import torch
print(torch.cuda.is_available())
print(torch.version.cuda)  # 如果 CUDA 可用，这将显示 CUDA 版本信息                                                            
print(torch.__version__)
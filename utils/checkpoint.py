import os
import shutil
import torch






def save_checkpoint(state, is_best, save_path):
    # is no folder
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, "checkpoint.pth")
    
    # 현재 checkpoint 저장
    torch.save(state, checkpoint_path)
    
    # save best model
    if is_best:
        best_path = os.path.join(save_path, "model_best.pth")
        shutil.copyfile(checkpoint_path, best_path)

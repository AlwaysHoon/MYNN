import os
import time
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from args import main_args
from utils import  setup_logger, save_checkpoint
from utils.engine import train, validate

# ================================================================
# Main
# ================================================================

def main():
    #----------------------------
    # 1. Device
    #----------------------------
    parser = main_args()
    args = parser.parse_args()
    logger = setup_logger(args.save_path)
    logger.info(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    #-----------------------------
    # 2. Data Processing
    #-----------------------------
    # ImageNet의 기본 normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Augmentation
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    val_transform   = transforms.Compose([transforms.Resize(256), transforms.CentorCrop(224), transforms.ToTensor(), normalize])
    
    # Data Processing
    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(args.data, "val"),   transform=val_transform)    
    
    #-----------------------------
    # 3. Data Loader
    #-----------------------------
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True,  num_workers = args.workers, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size = args.batch_size, shuffle = False, num_workers = args.workers, pin_memory=True)
    
    #-----------------------------
    # 4. Load Model
    #-----------------------------
    # torchvision에서 모델 생성
    model = models.get_model(args.arch, weights=None, num_classes = len(train_dataset.classes))
    model = model.to(device)
    logger.info(model)
    
    #-----------------------------
    # 5. Criterion / Optimizer
    #-----------------------------
    criterion = nn.CrossEntropyLoss
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.wd)
    
   
    # Pretrained / Resume checkpoint
    start_epoch = 0
    best_acc = 0.0
    
    if args.pretrained and not args.resume:
        logger.info(f"=> using pre-trained model: {args.arch}")
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location = device)
        model.load_state_dict(      checkpoint["model"])
        optimizer.load_state_dict(  checkpoint["optimizer"])
        start_epoch =               checkpoint["epoch"]
        best_acc    =               checkpoint["best_acc"]
        
        logger.info(f"Checkpoint loaded: {args.resume}")
        logger.info(f"resume epoch: {start_epoch}, last_best_acc: {best_acc}")
        
    #-----------------------------
    # 6. Training / Validation
    #-----------------------------
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch[{epoch+1}/{args.epochs}]")
    
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, logger)
        val_loss,   val_acc   = validate(val_loader, model, criterion, device, logger)
        logger.info(f"Train Loss: {train_loss:4f} | "
                    f"Train Acc : {train_acc:.2f}")
        logger.info(f"Val Loss  : {val_loss:4f} | "
                    f"Val Acc   : {val_acc:.2f}")
        
        #-----------------------------
        # 7. Save checkpoint
        #-----------------------------
        is_best = val_acc > best_acc
        
        if is_best:
            best_acc = val_acc
            logger.info(f"Best model | Epoch: {epoch} | Acc: {best_acc}")
        
        save_checkpoint({
            "epoch"    : epoch + 1,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc" : best_acc
            }, is_best, args.save_path)
    
    logger.info(f"Final Acc: {best_acc}")   



if __name__ == "__main__":
    main()
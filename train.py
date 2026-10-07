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
# from utils.engine import train, validate

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
    # ImageNet basic normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Augmentation
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    val_transform   = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    
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
    # get model from torchvision.model
    model = models.get_model(args.arch, weights=None, num_classes = len(train_dataset.classes))
    model = model.to(device)
    logger.info(model)
    
    #-----------------------------
    # 5. Criterion / Optimizer
    #-----------------------------
    criterion = nn.CrossEntropyLoss()
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
            logger.info(f"Best model | Epoch: {epoch + 1} | Acc: {best_acc}")
        
        save_checkpoint({
            "epoch"    : epoch + 1,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc" : best_acc
            }, is_best, args.save_path)
    
    logger.info(f"Final Acc: {best_acc}")   

def train(train_loader, model, criterion, optimizer, device, logger):
    
    # Dropout, BatchNorm 등을 train mode로 설정
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # data를 gpu로 이동
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 이전 iteration의 gradient 제거
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images) #shape: [batch_size,1000]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
    
        batch_size = targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
        
        total_loss += (loss.item()*images.size(0))
        total      += batch_size
        correct    += batch_correct

        batch_acc = (100.0 * batch_correct / batch_size)
        logger.info(
            f"Train Batch [{batch_idx + 1}/{len(train_loader)}] | "
            f"Loss : {loss.item():.4f} | "
            f"Acc : {batch_acc:.2f}"
        )
    average_loss = total_loss / total
    accuracy = (100.0 * correct / total)
    
    return average_loss, accuracy

# ================================================================
# Validation
# ================================================================
def validate(val_loader, model, criterion, device, logger):
    # Dropout, BatchNorm 등을 train mode로 설정
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    #validation에서는 gradient가 필요없음
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            # data를 gpu로 이동
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            
            batch_size = targets.size(0)
            batch_correct = predicted.eq(targets).sum().item()
                        
            total_loss += (loss.item()*images.size(0))
            total      += batch_size
            correct    += batch_correct
            
            batch_acc = (100.0 * batch_correct / batch_size)
            
            logger.info(
                f"Val Batch [{batch_idx+1}/{len(val_loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Acc : {batch_acc:.2f}"
                    )
            
    average_loss = total_loss / total
    accuracy = (100.0 * correct / total)
        
    return average_loss, accuracy


if __name__ == "__main__":
    main()
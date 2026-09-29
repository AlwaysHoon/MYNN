import torch
from .logger import setup_logger


# ================================================================
# Training
# ================================================================
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
        outputs = model(images)
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
            f"Train Batch [{batch_idx+1}/{len(train_loader)}] | "
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
                f"Train Batch [{batch_idx+1}/{len(val_loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Acc : {batch_acc:.2f}"
                    )
            
    average_loss = total_loss / total
    accuracy = (100.0 * correct / total)
        
    return average_loss, accuracy
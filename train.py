import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
import model, data, config
from utils import compute_loss
from model import ResNetYOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_root_dir = os.path.join(current_dir, 'jellyfish', 'Train_Test_Valid')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per GPU')
    parser.add_argument('--img-size', nargs='+', type=int, default=[224, 224], help='image sizes')
    
    # Hardware parameters
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    
    # Save parameters
    parser.add_argument('--project', default='runs/train', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=False, help='existing project/name ok')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    
    # Dataset parameters
    parser.add_argument('--root-dir', type=str, default=default_root_dir, help='root directory of dataset')
    parser.add_argument('--train-path', type=str, default='train', help='path to training data')
    parser.add_argument('--val-path', type=str, default='valid', help='path to validation data')
    
    # Additional parameters
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    
    opt = parser.parse_args()
    return opt

def train(hyp, opt, device):
    # Create output directory with unique name if it already exists
    save_dir = os.path.join(opt.project, opt.name)
    if os.path.exists(save_dir) and not opt.exist_ok:
        # Find the next available exp directory
        i = 0
        while True:
            i += 1
            new_dir = os.path.join(opt.project, f'exp{i}')
            if not os.path.exists(new_dir):
                save_dir = new_dir
                break
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    model = ResNetYOLO(num_classes=config.MODEL.num_classes, hyp=hyp)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), 
                         lr=hyp['lr'], 
                         momentum=hyp['momentum'], 
                         weight_decay=hyp['weight_decay'])
    
    # Initialize dataloaders
    train_loader = data.create_dataloader(
        os.path.join(opt.root_dir, opt.train_path),
        'train',
        batch_size=opt.batch_size
    )
    val_loader = data.create_dataloader(
        os.path.join(opt.root_dir, opt.val_path),
        'valid',
        batch_size=opt.batch_size,
        shuffle=False
    )
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=save_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(opt.epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        for batch_i, (images, targets) in enumerate(pbar):
            print(f"Targets shape: {targets.shape}")  # 디버깅용
            print(f"Targets dtype: {targets.dtype}")  # 디버깅용

            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute loss
            loss, loss_items = compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log batch results
            if batch_i % 10 == 0:
                writer.add_scalar('train/batch_loss', loss.item(), 
                                epoch * len(train_loader) + batch_i)
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        
        # Validation phase
        if not opt.noval:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc='Validating'):
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    
                    # Compute validation loss
                    loss, _ = compute_loss(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar('val/epoch_loss', avg_val_loss, epoch)
            
            # Save best model
            if avg_val_loss < best_val_loss and not opt.nosave:
                best_val_loss = avg_val_loss
                save_path = os.path.join(save_dir, 
                                       f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, save_path)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{opt.epochs}:")
            print(f"Training Loss: {avg_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
    writer.close()
    
    # Save final model if requested
    if not opt.nosave:
        final_save_path = os.path.join(save_dir, 
                                     f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        torch.save({
            'epoch': opt.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, final_save_path)

if __name__ == "__main__":
    # Parse command line arguments
    opt = parse_opt()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load hyperparameters from config
    hyp = config.CONFIG.hyp
    
    # Start training
    train(hyp, opt, device)
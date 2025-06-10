import torch
from torch.utils.data import DataLoader
from typing import Tuple
from app.src.utils.config import checkpoint_path
import time
from app.src.utils.model_manager import ModelManager

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device,
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.long).squeeze()
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
        return total_loss / len(train_loader), 100 * correct / total

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device, dtype=torch.long).squeeze()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
        return total_loss / len(val_loader), 100 * correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, patience: int):
        patience_counter = 0
        best_val_loss = float('inf')
        
        # counting time
        start_time = time.time()

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            if epoch == 0:
                yield "Training started...\n\n\n"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                print("Early stopping triggered!")
                ModelManager.update_model(new_model=self.model)
                break
            
        # returning the time taken for training
        end_time = time.time()
        yield f"Training completed in {end_time - start_time:.2f} seconds.\n\n\n"   
import torch
import numpy as np
import os


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True, 
                 save_dir='models', metric='loss', mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last time improvement occurred.
            min_delta (float): Minimum change in the monitored quantity to qualify as improvement.
            restore_best_weights (bool): Whether to restore model weights from the epoch 
                                       with the best value of the monitored quantity.
            save_dir (str): Directory to save the best model.
            metric (str): Metric to monitor ('loss', 'acc', 'val_loss', 'val_acc')
            mode (str): 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.save_dir = save_dir
        self.metric = metric
        self.mode = mode
        
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
    def __call__(self, current_value, model, epoch=None):
        """
        Check if training should be stopped early.
        
        Args:
            current_value (float): Current value of the monitored metric
            model (torch.nn.Module): Model to save
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check if current value is better than best value
        if self.mode == 'min':
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)
        
        if is_better:
            self.best_value = current_value
            self.counter = 0
            
            # Save best model weights
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                
            # Save best model to file
            best_model_path = os.path.join(self.save_dir, f'best_{self.metric}_model.pth')
            torch.save(model.state_dict(), best_model_path)
            
            if epoch is not None:
                info_file = os.path.join(self.save_dir, f'best_{self.metric}_info.txt')
                with open(info_file, 'w') as f:
                    f.write(f"Best {self.metric}: {self.best_value:.6f}\n")
                    f.write(f"Epoch: {epoch}\n")
                    f.write(f"Model: {best_model_path}\n")
                    
        else:
            self.counter += 1
            
        # Check if we should stop
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"Early stopping triggered. Best {self.metric}: {self.best_value:.6f}")
            
            # Restore best weights if requested
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print("Restored best model weights")
                
        return self.early_stop


class ModelCheckpoint:
    """
    Save the model when a monitored metric improves.
    """
    def __init__(self, filepath='best_model.pth', monitor='val_loss', mode='min',
                 save_best_only=True, verbose=1):
        """
        Args:
            filepath (str): Path to save the model file
            monitor (str): Quantity to monitor
            mode (str): 'min' or 'max'
            save_best_only (bool): If True, only save when the model is considered the best
            verbose (int): Verbosity level
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best = np.inf if mode == 'min' else -np.inf
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
    def __call__(self, current_value, model, optimizer=None, epoch=None, **kwargs):
        """
        Check if model should be saved.
        
        Args:
            current_value (float): Current value of monitored metric
            model (torch.nn.Module): Model to save
            optimizer: Optimizer state to save (optional)
            epoch (int): Current epoch
            **kwargs: Additional information to save
        """
        if self.mode == 'min':
            is_better = current_value < self.best
        else:
            is_better = current_value > self.best
            
        if not self.save_best_only or is_better:
            if is_better:
                self.best = current_value
                
            # Prepare checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_value': self.best,
                'current_value': current_value,
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                
            # Add any additional information
            checkpoint.update(kwargs)
            
            # Save checkpoint
            torch.save(checkpoint, self.filepath)
            
            if self.verbose > 0:
                print(f"Model saved to {self.filepath} ({self.monitor}: {current_value:.6f})")


class ClassBalancedMetrics:
    """
    Track metrics with special attention to minority classes.
    """
    def __init__(self, num_classes=None, class_names=None, minority_class=None):
        if num_classes is None:
            raise ValueError("必须指定 num_classes 参数!")
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        if minority_class is None:
            # 自动识别少数类别(假设索引最大的是少数类)
            minority_class = num_classes - 1
        self.minority_class = minority_class
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        
    def update(self, pred, target):
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred (torch.Tensor): Predictions
            target (torch.Tensor): Ground truth targets
        """
        if hasattr(pred, 'cpu'):
            pred = pred.cpu()
        if hasattr(target, 'cpu'):
            target = target.cpu()
            
        self.predictions.extend(pred.numpy() if hasattr(pred, 'numpy') else pred)
        self.targets.extend(target.numpy() if hasattr(target, 'numpy') else target)
        
    def compute(self):
        """
        Compute various metrics including class-specific accuracies.
        
        Returns:
            dict: Dictionary containing various metrics
        """
        if not self.predictions:
            return {}
            
        pred = np.array(self.predictions)
        target = np.array(self.targets)
        
        # Overall accuracy
        overall_acc = np.mean(pred == target)
        
        # Class-specific accuracies
        class_accs = {}
        class_counts = {}
        
        for i in range(self.num_classes):
            mask = target == i
            if np.sum(mask) > 0:
                class_accs[f'{self.class_names[i]}_acc'] = np.mean(pred[mask] == target[mask])
                class_counts[f'{self.class_names[i]}_count'] = np.sum(mask)
            else:
                class_accs[f'{self.class_names[i]}_acc'] = 0.0
                class_counts[f'{self.class_names[i]}_count'] = 0
                
        # Minority class specific metrics
        minority_acc = class_accs.get(f'{self.class_names[self.minority_class]}_acc', 0.0)
        
        # Balanced accuracy (average of class accuracies)
        valid_accs = [acc for acc in class_accs.values() if isinstance(acc, float)]
        balanced_acc = np.mean(valid_accs) if valid_accs else 0.0
        
        metrics = {
            'overall_accuracy': overall_acc,
            'balanced_accuracy': balanced_acc,
            'minority_class_accuracy': minority_acc,
            **class_accs,
            **class_counts
        }
        
        return metrics
        
    def get_minority_score(self):
        """Get a composite score focusing on minority class performance."""
        metrics = self.compute()
        minority_acc = metrics.get('minority_class_accuracy', 0.0)
        balanced_acc = metrics.get('balanced_accuracy', 0.0)
        
        # Weighted combination: 70% minority class accuracy + 30% balanced accuracy
        composite_score = 0.7 * minority_acc + 0.3 * balanced_acc
        return composite_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CreditDataset(Dataset):
    def __init__(self, X, y_main, y_protected):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y_main = torch.FloatTensor(y_main.values if hasattr(y_main, 'values') else y_main)
        
        # y_protected is now a 2D matrix handling multiple demographic columns
        self.y_protected = torch.FloatTensor(y_protected.values if hasattr(y_protected, 'values') else y_protected)
        
        # Failsafe: If passing only 1 protected attribute, ensure it remains a 2D column vector
        if len(self.y_protected.shape) == 1:
            self.y_protected = self.y_protected.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_main[idx], self.y_protected[idx]


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class AutoFairNetwork(nn.Module):
    def __init__(self, input_dim, num_protected):
        super(AutoFairNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Predicts Default (0 or 1)
        self.main_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        # Predicts ALL protected attributes simultaneously (e.g., Sex AND Education)
        self.adversary_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_protected) # Dynamic output size based on targets
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        main_logits = self.main_head(features)
        reversed_features = GradientReversalFn.apply(features, alpha)
        adv_logits = self.adversary_head(reversed_features)
        return main_logits, adv_logits


def train_adversarial_network(X_train, y_train, protected_train, epochs=30, batch_size=64):
    logger.info("--- ROUTE B: Training PyTorch Intersectional Adversary ---")
    
    # Determine how many protected attributes we are fighting simultaneously
    num_protected = protected_train.shape[1] if len(protected_train.shape) > 1 else 1
    logger.info(f"Adversary configured to penalize {num_protected} protected demographic(s) simultaneously.")

    dataset = CreditDataset(X_train, y_train, protected_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AutoFairNetwork(input_dim=X_train.shape[1], num_protected=num_protected)
    
    criterion_main = nn.BCEWithLogitsLoss()
    # Loss calculated across all demographic targets
    criterion_adv = nn.BCEWithLogitsLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_main_loss = 0
        total_adv_loss = 0
        
        p = float(epoch) / epochs
        alpha = 2. / (1. + np.exp(-10. * p)) - 1

        for X_batch, y_main, y_prot in train_loader:
            optimizer.zero_grad()
            main_logits, adv_logits = model(X_batch, alpha)
            
            # Squeeze safely for 1D main target
            loss_main = criterion_main(main_logits.squeeze(), y_main)
            
            # View forces shapes to align for multi-label adversary loss
            loss_adv = criterion_adv(adv_logits.view(-1, num_protected), y_prot.view(-1, num_protected))
            
            total_loss = loss_main + loss_adv
            total_loss.backward()
            optimizer.step()
            
            total_main_loss += loss_main.item()
            total_adv_loss += loss_adv.item()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] | Main Loss: {total_main_loss/len(train_loader):.4f} | Adv Loss: {total_adv_loss/len(train_loader):.4f}")
            
    return model


def predict_adversarial(model, X_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
        main_logits, _ = model(X_tensor)
        preds = (torch.sigmoid(main_logits.squeeze()) > 0.5).int().numpy()
    return preds
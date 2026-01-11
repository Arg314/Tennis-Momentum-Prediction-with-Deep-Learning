import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from .models import get_model
from .utils import get_device

def train_model(seq_X, scalar_X, y, config, model_name):
    train_cfg = config['train']
    device = get_device(train_cfg['device'])
    
    X_seq_tr, X_seq_val, X_scal_tr, X_scal_val, y_tr, y_val = train_test_split(
        seq_X, scalar_X, y,
        test_size=train_cfg['test_size'],
        random_state=train_cfg['random_state']
    )
    
    model = get_model(model_name, config).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    
    # Prepare data loaders
    if y.ndim == 1:
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    else:
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_seq_tr, dtype=torch.float32),
            torch.tensor(X_scal_tr, dtype=torch.float32),
            y_tr_tensor
        ),
        batch_size=train_cfg['batch_size'],
        shuffle=True
    )
    
    for epoch in range(train_cfg['epochs']):
        model.train()
        for seq_x, scal_x, y_batch in train_loader:
            seq_x, scal_x, y_batch = seq_x.to(device), scal_x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(seq_x, scal_x)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Save model
    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), f"trained_models/{model_name.lower()}_model.pth")
    return model, (X_seq_val, X_scal_val, y_val_tensor)
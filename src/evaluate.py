import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .utils import get_device

def evaluate_model(model, X_seq_val, X_scal_val, y_val_true, config):
    device = get_device(config['train']['device'])
    model.eval()
    
    with torch.no_grad():
        seq_tensor = torch.tensor(X_seq_val, dtype=torch.float32).to(device)
        scal_tensor = torch.tensor(X_scal_val, dtype=torch.float32).to(device)
        preds = model(seq_tensor, scal_tensor).cpu().numpy()
    
    if y_val_true.ndim == 2:  # Multi-step
        results = {}
        y_np = y_val_true.numpy()
        for step in range(y_np.shape[1]):
            acc = accuracy_score(y_np[:, step], preds[:, step] > 0.5)
            results[f"step_{step+1}_acc"] = acc
        return results
    else:
        y_flat = y_val_true.flatten()
        preds_flat = preds.flatten()
        acc = accuracy_score(y_flat, preds_flat > 0.5)
        f1 = f1_score(y_flat, preds_flat > 0.5)
        auc = roc_auc_score(y_flat, preds_flat)
        return {"acc": acc, "f1": f1, "auc": auc}
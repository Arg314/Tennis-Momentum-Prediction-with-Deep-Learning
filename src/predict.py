import torch
from .models import get_model
from .utils import get_device

def load_trained_model(model_name, config):
    device = get_device(config['train']['device'])
    model = get_model(model_name, config).to(device)
    model.load_state_dict(torch.load(f"trained_models/{model_name.lower()}_model.pth", map_location=device))
    model.eval()
    return model

def predict_next_point(model, seq_x, scal_x, config):
    device = get_device(config['train']['device'])
    with torch.no_grad():
        seq_tensor = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).to(device)
        scal_tensor = torch.tensor(scal_x, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(seq_tensor, scal_tensor).cpu().item()
    return pred
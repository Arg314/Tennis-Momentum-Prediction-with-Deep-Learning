import torch
import torch.nn as nn

class TennisLSTM(nn.Module):
    def __init__(self, seq_dim=3, scalar_dim=1, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, layers, batch_first=True)
        self.scalar_fc = nn.Linear(scalar_dim, hidden)
        self.fc = nn.Linear(hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, seq_x, scal_x):
        lstm_out, _ = self.lstm(seq_x)
        feat_seq = lstm_out[:, -1, :]
        feat_scal = self.scalar_fc(scal_x)
        out = self.fc(torch.cat([feat_seq, feat_scal], dim=1))
        return self.sigmoid(out)

class TennisRNN(nn.Module):
    def __init__(self, seq_dim=3, scalar_dim=1, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.RNN(seq_dim, hidden, layers, batch_first=True)
        self.scalar_fc = nn.Linear(scalar_dim, hidden)
        self.fc = nn.Linear(hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, seq_x, scal_x):
        rnn_out, _ = self.rnn(seq_x)
        feat_seq = rnn_out[:, -1, :]
        feat_scal = self.scalar_fc(scal_x)
        out = self.fc(torch.cat([feat_seq, feat_scal], dim=1))
        return self.sigmoid(out)

class TennisGRU(nn.Module):
    def __init__(self, seq_dim=3, scalar_dim=1, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, layers, batch_first=True)
        self.scalar_fc = nn.Linear(scalar_dim, hidden)
        self.fc = nn.Linear(hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, seq_x, scal_x):
        gru_out, _ = self.gru(seq_x)
        feat_seq = gru_out[:, -1, :]
        feat_scal = self.scalar_fc(scal_x)
        out = self.fc(torch.cat([feat_seq, feat_scal], dim=1))
        return self.sigmoid(out)

class TennisTransformer(nn.Module):
    def __init__(self, seq_dim=3, scalar_dim=1, d_model=32, nhead=2, layers=2):
        super().__init__()
        self.embed = nn.Linear(seq_dim, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder, num_layers=layers)
        self.scalar_fc = nn.Linear(scalar_dim, d_model)
        self.fc = nn.Linear(d_model * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, seq_x, scal_x):
        x = self.embed(seq_x)
        trans_out = self.trans(x)
        feat_seq = trans_out[:, -1, :]
        feat_scal = self.scalar_fc(scal_x)
        out = self.fc(torch.cat([feat_seq, feat_scal], dim=1))
        return self.sigmoid(out)

class MultiStepLSTM(nn.Module):
    def __init__(self, seq_dim=3, scalar_dim=1, hidden=64, layers=2, horizon=3):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, layers, batch_first=True)
        self.scalar_fc = nn.Linear(scalar_dim, hidden)
        self.fc = nn.Linear(hidden * 2, horizon)
        self.sigmoid = nn.Sigmoid()
    def forward(self, seq_x, scal_x):
        lstm_out, _ = self.lstm(seq_x)
        feat_seq = lstm_out[:, -1, :]
        feat_scal = self.scalar_fc(scal_x)
        out = self.fc(torch.cat([feat_seq, feat_scal], dim=1))
        return self.sigmoid(out)

def get_model(model_type, config):
    model_cfg = config['model']
    feat_cfg = config['feature']
    if model_type == "LSTM":
        return TennisLSTM(hidden=model_cfg['hidden_size'], layers=model_cfg['num_layers'])
    elif model_type == "RNN":
        return TennisRNN(hidden=model_cfg['hidden_size'], layers=model_cfg['num_layers'])
    elif model_type == "GRU":
        return TennisGRU(hidden=model_cfg['hidden_size'], layers=model_cfg['num_layers'])
    elif model_type == "Transformer":
        return TennisTransformer(
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            layers=model_cfg['num_layers']
        )
    elif model_type == "MultiStepLSTM":
        return MultiStepLSTM(
            hidden=model_cfg['hidden_size'],
            layers=model_cfg['num_layers'],
            horizon=feat_cfg['horizon']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
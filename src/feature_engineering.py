import numpy as np

def build_features(points, window_size=10, horizon=1, use_momentum=True):
    if len(points) <= window_size + horizon - 1:
        return None, None, None
    
    labels = [1 if p['point_char'] in ['S','A'] else 0 for p in points]
    serves = [p['server_id'] for p in points]
    is_ace_df = [1 if p['point_char'] in ['A','D'] else 0 for p in points]
    
    seq_X, scalar_X, y = [], [], []
    
    for i in range(window_size, len(labels) - horizon + 1):
        past_labels = labels[i-window_size:i]
        past_serves = serves[i-window_size:i]
        past_adf = is_ace_df[i-window_size:i]
        seq_feat = np.column_stack([past_labels, past_serves, past_adf])
        
        if use_momentum:
            recent_wins = sum(labels[max(0, i-3):i])
            momentum = recent_wins / min(3, i)
            scalar_feat = np.array([momentum])
        else:
            scalar_feat = np.array([0.0])
        
        target = labels[i:i+horizon] if horizon > 1 else labels[i]
        
        seq_X.append(seq_feat)
        scalar_X.append(scalar_feat)
        y.append(target)
    
    return np.array(seq_X), np.array(scalar_X), np.array(y)

def prepare_dataset(df, config):
    feat_cfg = config['feature']
    all_seq_X, all_scalar_X, all_y = [], [], []
    
    for _, row in df.iterrows():
        X_seq, X_scal, y = build_features(
            row['parsed_points'],
            window_size=feat_cfg['window_size'],
            horizon=feat_cfg['horizon'],
            use_momentum=feat_cfg['use_momentum']
        )
        if X_seq is not None:
            all_seq_X.append(X_seq)
            all_scalar_X.append(X_scal)
            all_y.append(y)
    
    if not all_seq_X:
        raise ValueError("No valid sequences generated.")
    
    seq_X = np.concatenate(all_seq_X, axis=0)
    scalar_X = np.concatenate(all_scalar_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    return seq_X, scalar_X, y
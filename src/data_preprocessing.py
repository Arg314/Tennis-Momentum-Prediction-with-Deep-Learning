import os
import pandas as pd
from .utils import ensure_dir

def parse_pbp_detailed(pbp_str, server1_first=True):
    points = []
    set_idx = 0
    game_idx = 0
    server_id = 0 if server1_first else 1
    is_tiebreak = False
    game_score = [0, 0]
    
    for char in str(pbp_str):
        if char == '.':
            set_idx += 1
            game_idx = 0
            game_score = [0, 0]
            server_id = 1 - server_id
            is_tiebreak = False
            continue
        elif char == ';':
            game_idx += 1
            game_score = [0, 0]
            if not is_tiebreak:
                server_id = 1 - server_id
            continue
        elif char == '/':
            server_id = 1 - server_id
            continue
        elif char in ['S', 'R', 'A', 'D']:
            points.append({
                'point_char': char,
                'server_id': server_id,
                'set_idx': set_idx,
                'game_idx': game_idx,
                'is_tiebreak': is_tiebreak,
                'score_in_game': tuple(game_score)
            })
            if char in ['S', 'A']:
                game_score[0] += 1
            else:
                game_score[1] += 1
            if not is_tiebreak and game_idx >= 12 and game_score[0] == game_score[1]:
                is_tiebreak = True
    return points

def load_and_clean_data(config):
    data_cfg = config['data']
    raw_dir = data_cfg['raw_dir']
    files = [
        os.path.join(raw_dir, data_cfg['archive_file']),
        os.path.join(raw_dir, data_cfg['current_file'])
    ]
    
    df_list = []
    for f in files:
        if os.path.exists(f):
            df_temp = pd.read_csv(f)
            if 'tour' in df_temp.columns:
                df_temp = df_temp[df_temp['tour'] == 'ATP']
            df_list.append(df_temp)
    
    df_raw = pd.concat(df_list, ignore_index=True)
    print(f"原始数据：{len(df_raw)} 场 ATP 比赛")
    
    df_raw['parsed_points'] = df_raw['pbp'].apply(
        lambda x: parse_pbp_detailed(str(x)) if pd.notna(x) else []
    )
    df_raw['n_points'] = df_raw['parsed_points'].apply(len)
    
    df_clean = df_raw[df_raw['n_points'] >= data_cfg['min_points']].copy()
    print(f"清洗后：{len(df_clean)} 场有效比赛")
    
    # 保存处理后数据
    processed_path = os.path.join(data_cfg['processed_dir'], 'cleaned_matches.parquet')
    ensure_dir(data_cfg['processed_dir'])
    df_clean.to_parquet(processed_path, index=False)
    
    return df_clean
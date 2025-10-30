# [ 這是 dqn_baseline/play.py 的內容 ]

# -----------------------------------------------------------------
# 這是「基準版 DQN」的 GIF 錄製程式
# -----------------------------------------------------------------

import imageio
from IPython.display import HTML
from IPython.display import display as ipy_display
import torch
import time 
import os
from google.colab import drive # 為了 Colab 存檔

# 從同一個資料夾導入我們的 class
from snake_game import SnakeGameAI
from agent import Agent

# -----------------------------------------------------------------
# 警告：這個檔案是設計在 Google Colab 中執行的
# 它需要安裝 imageio-ffmpeg 並掛載 GDrive
# -----------------------------------------------------------------

def record_ai_game_to_gif():
    # --- 0. 安裝 & 掛載 (Colab 專用) ---
    print("--- 正在安裝 imageio-ffmpeg... ---")
    os.system('pip install imageio imageio-ffmpeg')
    
    print("--- 正在掛載 Google Drive... ---")
    try:
        drive.mount('/content/drive', force_remount=True)
    except Exception as e:
        print(f"Drive 掛載失敗: {e}")
        return

    # --- 1. 從 Google Drive 讀取「舊的」進度檔 ---
    DRIVE_PATH = '/content/drive/MyDrive/Colab_AI_Snake/'
    CHECKPOINT_PATH = DRIVE_PATH + 'snake_checkpoint.pth' # 讀取舊檔名
    
    GIF_FILENAME = 'ai_snake_game_dqn_baseline.gif'
    GIF_SAVE_PATH = DRIVE_PATH + GIF_FILENAME
    
    print(f"--- 開始錄製「基準版 DQN」AI 表演 (加載 {CHECKPOINT_PATH}) ---")
    
    agent = Agent() # 導入基準版的 Agent
    
    # --- 2. 讀取大腦 ---
    try:
        checkpoint = torch.load(CHECKPOINT_PATH)
        if 'model_state_dict' in checkpoint:
            agent.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            agent.model.load_state_dict(checkpoint) 
        print(f"--- 成功加載「基準版 DQN」模型 ---")
    except FileNotFoundError:
        print(f"錯誤：在你的 Google Drive 上找不到進度檔 '{CHECKPOINT_PATH}'！")
        return
    except Exception as e:
        print(f"加載模型失敗: {e}")
        return
        
    agent.epsilon = -1 
    agent.n_games = 100 
    game = SnakeGameAI()
    frames = [] 
    
    print("AI 正在玩遊戲中... 請稍候 (正在錄製)...")
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        frame = game.get_frame()
        frame_rgb = np.transpose(frame, (1, 0, 2))
        frames.append(frame_rgb)
        if done:
            print(f"遊戲結束！ 最終分數: {score}")
            break
            
    print(f"--- 正在將 {len(frames)} 幀畫面轉存為 GIF 到 Google Drive... ---")
    
    imageio.mimsave(GIF_SAVE_PATH, frames, fps=15) 
    
    print(f"--- GIF {GIF_FILENAME} 儲存完畢！---")
    print(f"--- 檔案已儲存於: {GIF_SAVE_PATH} ---")
    print("你現在可以到你的 Google Drive 裡查看 `Colab_AI_Snake` 資料夾了！")

# 讓 .py 檔案被執行時自動跑
if __name__ == '__main__':
    record_ai_game_to_gif()
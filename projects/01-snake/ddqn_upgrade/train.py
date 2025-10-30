# [ 這是 ddqn_upgrade/train.py 的內容 ]

# -----------------------------------------------------------------
# 這是「升級版 DDQN」的訓練程式
# -----------------------------------------------------------------

import time
import os
import torch
import matplotlib.pyplot as plt
from IPython import display as py_display
from google.colab import drive # 為了 Colab 存檔

# 從同一個資料夾導入我們的 class
from snake_game import SnakeGameAI
from agent import Agent # 導入的是 DDQN 版的 Agent

# -----------------------------------------------------------------
# 警告：這個檔案是設計在 Google Colab 中執行的
# -----------------------------------------------------------------

def train():
    # --- 0. 掛載 Google Drive (Colab 專用) ---
    print("--- 正在掛載 Google Drive... ---")
    try:
        drive.mount('/content/drive')
    except Exception as e:
        print(f"Drive 掛載失敗: {e}")
        print("將無法儲存進度！")
        
    # --- 1. 設定 Google Drive 儲存路徑 ---
    DRIVE_PATH = '/content/drive/MyDrive/Colab_AI_Snake/'
    CHECKPOINT_PATH = DRIVE_PATH + 'snake_checkpoint_ddqn.pth' # 新的檔名
    
    os.makedirs(DRIVE_PATH, exist_ok=True)

    agent = Agent() # 導入 DDQN 版的 Agent
    game = SnakeGameAI()

    # --- 2. 嘗試從 Google Drive 載入舊進度 ---
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict']) # 也要讀取冷靜大腦
        agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.n_games = checkpoint['n_games']
        record = checkpoint['record']
        total_score = checkpoint['total_score']
        plot_scores = checkpoint['plot_scores']
        plot_mean_scores = checkpoint['plot_mean_scores']
        
        print(f"--- 成功從 Google Drive 加載 [升級版 DDQN] 進度！---")
        print(f"--- 目前已訓練 {agent.n_games} 局，最高紀錄 {record} 分 ---")
        
    except FileNotFoundError:
        print(f"--- 找不到 [升級版 DDQN] 進度檔，從頭開始訓練 ---")
    except Exception as e:
        print(f"--- 加載進度失敗: {e}，從頭開始訓練 ---")
    
    
    ROUNDS_PER_SESSION = 300
    UPDATE_TARGET_EVERY = 10 # 【DDQN 升級點】每 10 局 同步一次大腦
    target_games = agent.n_games + ROUNDS_PER_SESSION
    
    print(f"--- 開始訓練 [升級版 DDQN]！目標是從 {agent.n_games} 局練到 {target_games} 局 ---")
    time.sleep(3)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    while agent.n_games <= target_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # 【DDQN 升級點】定期同步「主要大腦」和「冷靜大腦」
            if agent.n_games % UPDATE_TARGET_EVERY == 0:
                agent._update_target_model()

            new_record = False
            if score > record:
                record = score
                new_record = True

            if agent.n_games % 20 == 0 or new_record:
                if new_record:
                    print(f"🎉 新紀錄！第 {agent.n_games} 局: {score} 分。儲存 DDQN 模型...")
                else:
                    print(f"--- 正在儲存 DDQN 進度到 Google Drive (局數 {agent.n_games}) ---")
                
                # 【DDQN 升級點】儲存進度時，兩個大腦都要存
                checkpoint = {
                    'model_state_dict': agent.model.state_dict(),
                    'target_model_state_dict': agent.target_model.state_dict(),
                    'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
                    'n_games': agent.n_games,
                    'record': record,
                    'total_score': total_score,
                    'plot_scores': plot_scores,
                    'plot_mean_scores': plot_mean_scores,
                }
                torch.save(checkpoint, CHECKPOINT_PATH)

            print(f'局數: {agent.n_games}/{target_games}, 分數: {score}, 最高紀錄: {record}')

            # --- 更新分數圖表 (Colab 專用) ---
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games 
            plot_mean_scores.append(mean_score)
            
            ax.clear()
            ax.set_title('AI 訓練進度 [升級版 DDQN]')
            ax.set_xlabel('局數 (Games)')
            ax.set_ylabel('分數 (Score)')
            ax.plot(plot_scores, label='當局分數')
            ax.plot(plot_mean_scores, label='平均分數', linestyle='--')
            ax.legend(loc='upper left')
            if len(plot_scores) > 1:
                ax.text(len(plot_scores)-1, score, str(score))
                ax.text(len(plot_mean_scores)-1, mean_score, f'{mean_score:.2f}')
            py_display.display(fig)
            py_display.clear_output(wait=True)

    print(f"--- 本次 [升級版 DDQN] 訓練完成！ (共 {ROUNDS_PER_SESSION} 局) ---")
    print(f"--- AI 總共已訓練 {agent.n_games} 局 ---")
    if 'mean_score' in locals():
        print(f"最終平均分數: {mean_score:.2f}, 最高紀錄: {record}")
    else:
        print(f"最高紀錄: {record}")
    print(f"最終 DDQN 模型已儲存於: {CHECKPOINT_PATH}")
    plt.ioff()
    plt.show()

# 讓 .py 檔案被執行時自動跑 train()
if __name__ == '__main__':
    train()
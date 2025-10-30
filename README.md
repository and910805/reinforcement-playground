# 🎡 Reinforcement Playground — 強化學習遊樂園

歡迎來到我的 **強化學習實驗樂園**！  
這裡是我紀錄與展示各種 **Reinforcement Learning (RL)** 專案的地方，  
從 🐍「貪吃蛇」到 🐭「老鼠走迷宮」，每一個都是一次「AI 學會玩」的冒險。

---

## 🚀 專案理念：人人都能玩強化學習

我希望讓強化學習變得更輕鬆、可重現：  
💡 **Colab 友善設計** — 免費版 Google Colab 就能跑！  
🧠 **演算法導向** — 專注於策略學習，而非硬體堆砌。  
🎮 **互動與樂趣** — 看見 AI 從「亂動」到「學聰明」的過程。

---

## 🏗️ 專案架構

```bash
reinforcement-playground/
│
├── .gitignore
│
├── README.md           # ⭐️ 遊樂園大廳 (Lobby)
│
└── projects/
    │
    └── 01-snake/       # 🐍 貪吃蛇「展館」
        │
        ├── README.md   # ⭐️⭐️⭐️ 這是最重要的「A/B 比較報告」！
        │
        ├── dqn_baseline/     # 展區 A: 基準版 (舊的DQN)
        │   │
        │   ├── snake_game.py   # (步驟 3 程式碼)
        │   ├── model.py        # (步驟 4 程式碼)
        │   ├── agent.py        # (步驟 4 程式碼)
        │   ├── train.py        # (步驟 5 修正版 - 存檔在 dqn.pth)
        │   ├── play.py         # (步驟 7 修正版 - 讀取 dqn.pth)
        │   └── requirements.txt
        │
        └── ddqn_upgrade/     # 展區 B: 升級版 (新的DDQN)
            │
            ├── snake_game.py   # (步驟 3 程式碼)
            ├── model.py        # (步驟 4 DDQN 升級版)
            ├── agent.py        # (步驟 4 DDQN 升級版)
            ├── train.py        # (步驟 5 DDQN 升級版)
            ├── play.py         # (步驟 7 DDQN 升級版)
            └── requirements.txt
```

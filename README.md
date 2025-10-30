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
├── README.md              # 🎡 遊樂園「大廳」(你現在所在的地方)
├── .gitignore             # 🧹 忽略暫存檔與模型
│
└── projects/              # 各個展館 (子專案)
    ├── 01-snake-ddqn/     # 🐍 貪吃蛇強化學習
    │   ├── snake_game.py
    │   ├── model.py
    │   ├── agent.py
    │   ├── train.py
    │   ├── play.py
    │   └── README.md
    │
    └── 02-mouse-maze/     # 🐭 老鼠走迷宮 (Coming Soon)

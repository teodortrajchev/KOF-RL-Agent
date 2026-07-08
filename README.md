# KOF Wing RL Agent

> Train a Reinforcement Learning agent to play **King of Fighters Wing**  
> using PPO, OpenCV, and Stable-Baselines3 — for a university RL project.

---

## Project Structure

```
kof_rl/
│
├── env/
│   ├── __init__.py     # makes env/ a Python package
│   ├── kof_env.py      # Gymnasium environment (main glue)
│   ├── vision.py       # screen capture + frame preprocessing
│   ├── controls.py     # keyboard action mapper
│   └── rewards.py      # reward computation
│
├── train.py            # PPO training script
├── test.py             # evaluation script (load + run saved model)
├── requirements.txt    # all dependencies
└── README.md
```

---

## Quick Start (Windows 11)

### 1. Get the game running

1. Download **Ruffle** desktop: https://ruffle.rs/#downloads
2. Download **KOF Wing 1.0** as a `.swf` file
3. Open Ruffle → load the `.swf` → confirm the game runs
4. Note the window position (you'll need it for `vision.py`)

### 2. Set up Python

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If torch fails, install CPU version first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **Important:** Run your terminal as **Administrator** — `pydirectinput` needs it.

### 3. Configure the screen region

Edit `env/vision.py` and update `DEFAULT_GAME_REGION` to match  
where the Ruffle window sits on your screen:

```python
DEFAULT_GAME_REGION = {
    "top":    100,   # Y pixel of the top edge of the game window
    "left":   100,   # X pixel of the left edge
    "width":  800,   # width of the game window
    "height": 600,   # height of the game window
}
```

Find your coordinates:
```bash
python -c "import mss; s=mss.mss(); print(s.monitors)"
```

### 4. Test each module independently

```bash
# Test screen capture (opens live preview — press Q to close)
python -m env.vision

# Test keyboard controls (focus the game first — waits 3 seconds)
python -m env.controls

# Test reward logic (no game needed)
python -m env.rewards

# Test full environment (20 random steps)
python -m env.kof_env
```

### 5. Train

```bash
python train.py
```

Monitor in TensorBoard:
```bash
tensorboard --logdir ./logs
# open http://localhost:6006
```

### 6. Evaluate a trained model

```bash
python test.py --model checkpoints/kof_ppo_final
```

---

## Common Windows Issues

| Problem | Fix |
|---|---|
| Keys not registering | Run terminal as Administrator |
| Black screen capture | Make sure the game window is not minimised |
| `mss` wrong region | Re-measure window position; use `mss.monitors` |
| `pydirectinput` import error | `pip install pydirectinput` in the venv |
| `torch` install fails | Use CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `check_env` warnings | Safe to ignore in v1; will fix in later iterations |

---

## Roadmap

| Version | Feature |
|---|---|
| **v1 (now)** | Random actions, placeholder rewards, basic screen capture |
| v2 | HP bar detection with HSV masking, dense reward shaping |
| v3 | Frame stacking (4 frames), CNN policy |
| v4 | Combo learning, action sequences |
| v5 | Self-play, curriculum learning |

---

## How It Works

```
PPO Agent
   │ action (0-8)
   ▼
GameController (controls.py)
   │ DirectInput keypress
   ▼
KOF Wing in Ruffle (browser game)
   │ updated game screen
   ▼
ScreenCapture (vision.py)
   │ 84×84 greyscale frame
   ▼
KOFEnv (kof_env.py)
   │ obs, reward, done
   ▼
PPO Agent (learns)
```

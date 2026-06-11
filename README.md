# Agent System Complex Version

MuJoCo-based embodied agent demo with frontier exploration, SAC navigation, lidar mapping, RGB-D landmark detection, topological memory, and a Qt dashboard.

## Demo

[![Demo video preview](assets/demo-preview.png)](https://cdn.jsdelivr.net/gh/BigWhiteCPN/mujoco-humanoid-hierarchical-rl-llm-spatial-navigation@main/assets/demo.mp4)

[Watch the 2:45 demo video](https://cdn.jsdelivr.net/gh/BigWhiteCPN/mujoco-humanoid-hierarchical-rl-llm-spatial-navigation@main/assets/demo.mp4) ([WebM fallback](https://cdn.jsdelivr.net/gh/BigWhiteCPN/mujoco-humanoid-hierarchical-rl-llm-spatial-navigation@main/assets/demo.webm)).

## Run

The repository includes the runtime assets used by the demo:

- MuJoCo robot XML and meshes under `resources/`
- Low-level locomotion policy at `models/policy_20251026.pt`
- SAC navigation model at `models/sac_lidar_interrupted_good3_0.91.zip`
- Random-map base environment at `visual_train/robot_visual_env_random_map.py`

Create a local `.env` or export `SILICONFLOW_API_KEY` before running LLM-driven commands:

```bash
cp .env.example .env
# edit .env and set SILICONFLOW_API_KEY
```

The bundled assets are used by default. Override these only if you want to test different files:

```bash
export ROBOT_MODEL_XML="resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302_copy.xml"
export LOW_LEVEL_POLICY_PATH="models/policy_20251026.pt"
export SAC_MODEL_PATH="models/sac_lidar_interrupted_good3_0.91.zip"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

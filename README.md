# Agent System Complex Version

MuJoCo-based embodied agent demo with frontier exploration, SAC navigation, lidar mapping, RGB-D landmark detection, topological memory, and a Qt dashboard.

## Demo

[![Demo video preview](assets/demo-preview.png)](assets/demo.mp4)

[Watch the 2:45 demo video](assets/demo.mp4) ([WebM fallback](assets/demo.webm)).

## Notes

This folder is intended to be used inside the original IsaacLabExtensionTemplate workspace. Runtime assets are not committed:

- MuJoCo robot XML, configured by `ROBOT_MODEL_XML`
- Low-level locomotion policy, configured by `LOW_LEVEL_POLICY_PATH`
- SAC navigation model, configured by `SAC_MODEL_PATH`
- Runtime memory logs under `memory_logs/`

Create a local `.env` or export environment variables before running:

```bash
export SILICONFLOW_API_KEY="your_api_key_here"
export ROBOT_MODEL_XML="/absolute/path/to/robot.xml"
export LOW_LEVEL_POLICY_PATH="/absolute/path/to/policy.pt"
export SAC_MODEL_PATH="/absolute/path/to/sac_model.zip"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

The code also depends on `robot_visual_env_random_map.py` from the sibling `scripts/visual_train/` directory in the original workspace.

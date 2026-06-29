# Asset Inventory

This repository includes the runtime assets needed by the demo so a reviewer can run it without reconstructing the training setup.

## Policies

- `models/policy_20251026.pt`
  - Low-level locomotion policy used by the MuJoCo robot controller.
  - Expected by `AgentVisualEnv` through `LOW_LEVEL_POLICY_PATH`.

- `models/sac_lidar_interrupted_good3_0.91.zip`
  - Stable-Baselines3 SAC policy for local navigation over lidar/grid observations.
  - Expected by `NavigationSkill` through `SAC_MODEL_PATH`.

## Robot And Scene Assets

- `resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302_copy.xml`
  - Main MuJoCo XML used by the demo.

- `resources/meshes/*.STL`
  - Robot mesh files referenced by the MuJoCo XML.

- `resources/meshes/furniture_chair_kenney.obj`
- `resources/meshes/furniture_chair_kenney.mtl`
  - Furniture mesh used as a scene object.
  - Keep the original attribution and license with the asset if it is replaced or redistributed outside this demo.

## Media

- `assets/demo-preview.png`
- `assets/demo.mp4`
- `assets/demo.webm`

These files are used only for the README preview and demo playback.

## Runtime Outputs

The following are generated at runtime and should not be committed:

- `memory_logs/`
- `*.npy`, `*.npz`
- `*.log`, `*.out`
- MuJoCo crash/log files such as `MJDATA.TXT` and `MUJOCO_LOG.TXT`

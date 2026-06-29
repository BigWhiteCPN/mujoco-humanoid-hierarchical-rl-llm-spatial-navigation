from scripts.check_demo_assets import check_assets


def test_check_assets_reports_missing_files(tmp_path):
    missing, empty = check_assets(tmp_path)

    assert missing
    assert not empty


def test_check_assets_accepts_minimal_valid_tree(tmp_path):
    required_files = [
        "resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302_copy.xml",
        "models/policy_20251026.pt",
        "models/sac_lidar_interrupted_good3_0.91.zip",
        "visual_train/robot_visual_env_random_map.py",
        "assets/demo-preview.png",
    ]

    for relative_path in required_files:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    mesh_dir = tmp_path / "resources/meshes"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "base_link.STL").write_text("mesh", encoding="utf-8")

    missing, empty = check_assets(tmp_path)

    assert missing == []
    assert empty == []

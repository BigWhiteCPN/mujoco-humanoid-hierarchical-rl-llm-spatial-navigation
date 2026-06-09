import time
import os

import numpy as np
import torch
from PIL import Image


class SegFormerSemanticSegmenter:
    """Lazy SegFormer ADE20K semantic segmentation for dashboard overlays."""

    def __init__(
        self,
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        min_interval_s=0.6,
        inference_size=(384, 384),
    ):
        self.model_name = model_name
        self.min_interval_s = min_interval_s
        self.inference_size = inference_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = None
        self.model = None
        self.id2label = {}
        self.wall_ids = set()
        self.floor_ids = set()

        self.last_run_time = 0.0
        self.last_overlay = None
        self.last_facts = {
            "wall_ratio": 0.0,
            "floor_ratio": 0.0,
            "status": "not_initialized",
        }
        self.failed = False

    def _ensure_model(self):
        if self.failed:
            return False
        if self.model is not None:
            return True

        try:
            self._normalize_proxy_env()
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

            print(f"[语义视觉] 正在加载 SegFormer: {self.model_name}")
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
                self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.id2label = {
                int(k): str(v).lower()
                for k, v in self.model.config.id2label.items()
            }
            self.wall_ids = {
                idx for idx, label in self.id2label.items()
                if "wall" in label
            }
            self.floor_ids = {
                idx for idx, label in self.id2label.items()
                if "floor" in label
            }
            print(
                f"[语义视觉] 模型加载完成: wall_ids={sorted(self.wall_ids)}, "
                f"floor_ids={sorted(self.floor_ids)}, device={self.device}"
            )
            return True
        except Exception as exc:
            self.failed = True
            self.last_facts = {
                "wall_ratio": 0.0,
                "floor_ratio": 0.0,
                "status": f"load_failed: {exc}",
            }
            print(f"[语义视觉] 加载失败，已自动禁用: {exc}")
            return False

    @staticmethod
    def _normalize_proxy_env():
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            value = os.environ.get(key)
            if value and value.startswith("socks://"):
                os.environ[key] = "socks5://" + value[len("socks://"):]

    def process(self, rgb_image, force=False):
        now = time.time()
        if (
            not force
            and self.last_overlay is not None
            and now - self.last_run_time < self.min_interval_s
        ):
            return self.last_overlay, self.last_facts

        if not self._ensure_model():
            self.last_overlay = rgb_image
            return rgb_image, self.last_facts

        try:
            image = Image.fromarray(np.asarray(rgb_image, dtype=np.uint8))
            original_size = image.size[::-1]
            image_for_model = image.resize(self.inference_size, Image.BILINEAR)
            inputs = self.processor(images=image_for_model, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            seg = logits.argmax(dim=1)[0].detach().cpu().numpy()

            wall_mask = np.isin(seg, list(self.wall_ids)) if self.wall_ids else np.zeros_like(seg, dtype=bool)
            floor_mask = np.isin(seg, list(self.floor_ids)) if self.floor_ids else np.zeros_like(seg, dtype=bool)

            overlay = np.asarray(rgb_image, dtype=np.float32).copy()
            overlay[floor_mask] = overlay[floor_mask] * 0.45 + np.array([0, 220, 80], dtype=np.float32) * 0.55
            overlay[wall_mask] = overlay[wall_mask] * 0.45 + np.array([255, 80, 20], dtype=np.float32) * 0.55
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            total = max(seg.size, 1)
            self.last_facts = {
                "wall_ratio": float(np.sum(wall_mask) / total),
                "floor_ratio": float(np.sum(floor_mask) / total),
                "status": "ok",
            }
            self.last_overlay = overlay
            self.last_run_time = now
            return overlay, self.last_facts
        except Exception as exc:
            self.last_facts = {
                "wall_ratio": 0.0,
                "floor_ratio": 0.0,
                "status": f"inference_failed: {exc}",
            }
            self.last_overlay = rgb_image
            return rgb_image, self.last_facts

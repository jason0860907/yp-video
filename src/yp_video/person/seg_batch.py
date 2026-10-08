"""Dense RF-DETR Seg over frame spans: the fp16 batch detector and the
decode-ahead producer that feeds it.

Shared by tracking (tracklets/tracking.py, Medium) and the person-head
labels pass (person/dense.py, 2XLarge). The producer thread decodes and
preprocesses frames (~9 ms/frame) while the GPU runs fixed-size batches, so a
dense pass is bound by the detector, not by decoding.
"""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from typing import NamedTuple

from yp_video.person.seg import PERSON_CLASS_ID

# Producer→consumer buffer (in frames). Small: it only needs to bridge the
# jitter between decode and inference, not hold a rally.
_QUEUE_FRAMES = 64


class BatchSegDetector:
    """fp16 batch-compiled RF-DETR Seg for a dense pass — person boxes,
    scores and instance masks in one forward (Medium: ~14.5 ms/frame at res
    432 on the 4090; the dense pass is bound by this, not by decoding).

    Separate from PersonDetector on purpose: optimize_for_inference() halves
    latency, and the compiled graph only accepts exactly ``batch_size``
    pre-resized tensors.
    """

    def __init__(self, variant: str, batch_size: int):
        #: An ``rfdetr`` Seg class name, e.g. "RFDETRSegMedium".
        self.variant = variant
        self.batch_size = batch_size
        self._model = None
        self.resolution: int | None = None

    def ensure(self) -> None:
        if self._model is not None:
            return
        import rfdetr
        import torch

        model = getattr(rfdetr, self.variant)()
        self.resolution = model.model_config.resolution
        model.optimize_for_inference(dtype=torch.float16, batch_size=self.batch_size)
        self._model = model

    def release(self) -> None:
        """Free the compiled model's VRAM (Medium ~8 GB at batch 16). The
        next ensure() rebuilds it."""
        if self._model is None:
            return
        import gc

        import torch

        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def predict_boxes(self, tensors: list, threshold: float) -> list:
        """``predict_batch`` without the masks: ``(xyxy, scores)`` numpy pairs
        per tensor, person class only, boxes in resolution-pixel space.

        rfdetr's ``predict`` always post-processes masks — every kept query
        upsampled to the input resolution and copied to the host — which at
        2XLarge's 768 px is most of a dense pass. This runs the same compiled
        graph and the same post-processor, handed the box outputs alone.
        """
        import torch
        import torchvision.transforms.functional as F

        model = self._model
        n = len(tensors)
        padded = tensors + [tensors[-1]] * (self.batch_size - n)
        batch = torch.stack([t.to(model.model.device) for t in padded])
        batch = F.normalize(batch, model.means, model.stds)
        with torch.no_grad():
            raw = model.model.inference_model(batch.to(dtype=model._optimized_dtype))
            sizes = torch.tensor([[self.resolution, self.resolution]] * len(padded), device=model.model.device)
            results = model.model.postprocess({"pred_logits": raw[1], "pred_boxes": raw[0]}, target_sizes=sizes)
        out = []
        for result in results[:n]:
            keep = (result["scores"] > threshold) & (result["labels"] == PERSON_CLASS_ID)
            out.append((result["boxes"][keep].float().cpu().numpy(), result["scores"][keep].float().cpu().numpy()))
        return out

    def predict_batch(self, tensors: list, threshold: float) -> list:
        """≤batch_size preprocessed (C, res, res) tensors → sv.Detections each
        (person class only, masks included), boxes in resolution-pixel space
        (callers scale back to frame pixels)."""
        n = len(tensors)
        padded = tensors + [tensors[-1]] * (self.batch_size - n)
        out = self._model.predict(padded, threshold=threshold, include_source_image=False)
        return [det[det.class_id == PERSON_CLASS_ID] for det in out[:n]]




class SpanFrame(NamedTuple):
    rally_id: int
    #: cv2's native frame index.
    frame: int
    #: The frame's presentation time (cv2 CAP_PROP_POS_MSEC), in seconds.
    time: float
    #: (C, res, res) float in [0, 1].
    tensor: object


@contextmanager
def span_frames(cap, spans: list[tuple[int, int, int]], *, stride: int, resolution: int, name: str):
    """Yield a ``SpanFrame`` per frame of the spans, decoded and
    preprocessed on a producer thread so the GPU never waits on ffmpeg or cv2.

    Frame indices are cv2's (seek + grab). INTER_AREA tracks torchvision's
    antialiased downscale (matched-detection IoU 0.99 vs the in-model resize
    path). A decode error surfaces from the iterator after the frames before it.
    """
    import cv2
    import torch

    frame_q: queue.Queue = queue.Queue(maxsize=_QUEUE_FRAMES)
    stop = threading.Event()
    producer_error: list[BaseException] = []

    def _put(item) -> bool:
        while not stop.is_set():
            try:
                frame_q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for rally_id, f0, f1 in spans:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
                for frame_idx in range(f0, f1 + 1):
                    if stop.is_set() or not cap.grab():
                        break
                    if (frame_idx - f0) % stride:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    rgb = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (resolution, resolution),
                        interpolation=cv2.INTER_AREA,
                    )
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
                    time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if not _put(SpanFrame(rally_id, frame_idx, time, tensor)):
                        return
                if stop.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 — surfaced to the consumer
            producer_error.append(exc)
        finally:
            _put(None)

    def frames():
        while (item := frame_q.get()) is not None:
            yield item
        if producer_error:
            raise producer_error[0]

    producer = threading.Thread(target=produce, name=name, daemon=True)
    producer.start()
    try:
        yield frames()
    finally:
        stop.set()
        producer.join(timeout=5)

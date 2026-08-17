from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Mapping, Sequence
import cv2
import numpy as np

OPERATIONS = ("gamma", "exposure", "contrast", "saturation", "rgb_gain", "shadow", "highlight",
              "directional_band", "bloom", "blur", "motion_blur", "noise")

@dataclass(frozen=True)
class CorruptionStats:
    dark_fraction: float
    white_fraction: float
    clipped_fraction: float
    operations: tuple[str, ...]

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class CorruptionResult:
    image: np.ndarray
    stats: CorruptionStats

@dataclass(frozen=True)
class CorruptionConfig:
    operation_probabilities: Mapping[str, float] = field(default_factory=lambda: {op: 1.0 for op in OPERATIONS})
    parameter_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        unknown=set(self.operation_probabilities)-set(OPERATIONS)
        if unknown or any(not np.isfinite(v) or not 0 <= v <= 1 for v in self.operation_probabilities.values()):
            raise ValueError("invalid operation probabilities")
        if any(len(v)!=2 or not np.isfinite(v).all() or v[0]>v[1] for v in (np.asarray(x,float) for x in self.parameter_ranges.values())):
            raise ValueError("invalid parameter ranges")

def plausibility_stats(image: np.ndarray, operations=()) -> CorruptionStats:
    return CorruptionStats(float(np.mean(image <= 5)), float(np.mean(image >= 250)),
                           float(np.mean((image == 0) | (image == 255))), tuple(operations))

def _field(shape, rng):
    h, w = shape[:2]
    small = rng.random((max(2, h // 16), max(2, w // 16))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

def generate_light_corruption(image: np.ndarray, rng: np.random.Generator, *, severity: int = 3,
                              probability: float = 0.4, operations: Sequence[str] | None = None,
                              config: CorruptionConfig | None = None) -> CorruptionResult:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 HxWx3 RGB")
    if severity not in range(1, 6):
        raise ValueError("severity must be an integer from 1 through 5")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    if not np.isfinite(image.astype(np.float32)).all():
        raise ValueError("image contains non-finite values")
    if np.mean((image == 0) | (image == 255)) > .98:
        raise ValueError("input image is implausibly clipped")
    config = config or CorruptionConfig()
    if rng.random() >= probability:
        copy = image.copy()
        return CorruptionResult(copy, plausibility_stats(copy))
    if operations is not None:
        ops=tuple(operations)
        if not 1 <= len(ops) <= 3 or any(op not in OPERATIONS for op in ops): raise ValueError("operations must name one to three known families")
    else:
        eligible=[op for op in OPERATIONS if rng.random() < config.operation_probabilities.get(op, 0.0)]
        if not eligible:
            copy=image.copy()
            return CorruptionResult(copy,plausibility_stats(copy))
        count=min(len(eligible),int(rng.integers(1,4)))
        ops=tuple(str(x) for x in rng.choice(eligible,size=count,replace=False))
    x = image.astype(np.float32)
    h, w = x.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    strength = severity / 5
    def draw(name, default):
        lo, hi = config.parameter_ranges.get(name, default)
        return rng.uniform(lo, hi)
    for op in ops:
        if op == "gamma": x = 255 * np.power(np.clip(x / 255, 0, 1), draw("gamma",(.65,1.45)) ** strength)
        elif op == "exposure": x *= 1+(draw("exposure",(.65,1.55))-1)*strength
        elif op == "contrast": x = (x - 127.5) * (1+(draw("contrast",(.55,1.5))-1)*strength) + 127.5
        elif op == "saturation":
            gray=np.mean(x,axis=2,keepdims=True); x=gray+(x-gray)*(1+(draw("saturation",(.4,1.6))-1)*strength)
        elif op == "rgb_gain": x *= rng.uniform(1 - .25 * strength, 1 + .25 * strength, (1, 1, 3))
        elif op == "shadow": x *= (1 - .55 * strength * _field(x.shape, rng)[..., None])
        elif op == "highlight":
            cx, cy = rng.uniform(0, w), rng.uniform(0, h); sx, sy = rng.uniform(.05*w,.3*w), rng.uniform(.04*h,.22*h)
            x += np.exp(-(((xx-cx)/sx)**2 + ((yy-cy)/sy)**2)/2)[...,None] * 150 * strength
        elif op == "directional_band":
            angle=rng.uniform(0,np.pi); offset=rng.uniform(-w,w); width=rng.uniform(.05*w,.25*w)
            band=np.exp(-((xx*np.cos(angle)+yy*np.sin(angle)-offset)/width)**2)
            x += band[...,None]*120*strength
        elif op == "bloom":
            clipped=np.maximum(x-(230-50*strength),0); x += cv2.GaussianBlur(clipped,(0,0),1+3*strength)
        elif op == "blur": x=cv2.GaussianBlur(x,(3,3),.3+1.2*strength)
        elif op == "motion_blur":
            k=3 if severity < 4 else 5; kernel=np.zeros((k,k),np.float32); kernel[k//2,:]=1/k; x=cv2.filter2D(x,-1,kernel)
        elif op == "noise": x += rng.normal(0, 1+8*strength, x.shape)
    out=np.clip(x,0,255).astype(np.uint8)
    stats=plausibility_stats(out,ops)
    if max(stats.dark_fraction,stats.white_fraction,stats.clipped_fraction) > .98:
        raise ValueError("generated corruption is implausible")
    return CorruptionResult(out, stats)

def random_light_corruption(image, rng, severity=3, probability=1.0):
    return generate_light_corruption(image, rng, severity=severity, probability=probability).image

def augment_light(image, seed, severity=3, probability=1.0):
    return random_light_corruption(image, np.random.default_rng(seed), severity, probability)

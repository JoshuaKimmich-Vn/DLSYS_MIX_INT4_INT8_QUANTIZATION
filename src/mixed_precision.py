"""Optimized mixed INT4/INT8 quantization-aware training pipeline."""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.ao.quantization import disable_observer, enable_observer
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wfdb

# ============================================================
# CONFIGURATION
# ============================================================
SIGNAL_LENGTH = 260
BATCH_SIZE = 64
EPOCHS = 400
LEARNING_RATE = 1e-3
VAL_RATIO = 0.1
TEST_RATIO = 0.3
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_WEIGHTS = [1.5, 1.0, 1.0, 1.0, 1.0]
EARLY_STOPPING_PATIENCE = 70
EARLY_STOPPING_MIN_DELTA = 0.001
LR_SCHEDULER_PATIENCE = 15
LR_SCHEDULER_FACTOR = 0.5
LR_MIN = 1e-6

GROUP_ID = "NoSMOTE_noBN_MixedPrecision"

# QAT config
QAT_EPOCHS = 50
QAT_LR = 5e-05
QAT_WARMUP_EPOCHS = 5
QAT_GRAD_CLIP = 1.0

# Mixed precision strategy
SENSITIVITY_RATIO_4BIT = 0.4

# Repository-relative paths. Override the dataset location with MITBIH_DATASET_PATH.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QAT_REPORT_PATH = os.path.join(PROJECT_ROOT, "config", "int8_sensitivity.json")
BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_fp32.pth")
QAT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mixed_qat_best.pth")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "generated")
DATASET_PATH = os.environ.get(
    "MITBIH_DATASET_PATH", os.path.join(PROJECT_ROOT, "data", "mitdb")
)

# ============================================================
# UTILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())


def normalize_layer_name(name):
    """Normalize legacy Sequential names such as ``pointwise.0``."""
    return name[:-2] if name.endswith(".0") else name


def estimate_deployment_sizes(model, four_bit_layers):
    """Estimate packed model sizes using a consistent raw-byte basis."""
    four_bit_names = {normalize_layer_name(name) for name in four_bit_layers}
    result = {
        "fp32_raw_bytes": 0,
        "full_int8_packed_bytes": 0,
        "mixed_packed_bytes": 0,
        "n_4bit_weights": 0,
        "n_8bit_weights": 0,
        "bias_params": 0,
    }
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv1d, nn.Linear)):
            continue
        weight_count = module.weight.numel()
        bias_count = module.bias.numel() if module.bias is not None else 0
        output_channels = module.weight.shape[0]
        metadata_bytes = output_channels * 8 + 8
        bias_bytes = bias_count * 4

        result["fp32_raw_bytes"] += (weight_count + bias_count) * 4
        result["full_int8_packed_bytes"] += weight_count + metadata_bytes + bias_bytes
        if name in four_bit_names:
            result["mixed_packed_bytes"] += (weight_count + 1) // 2
            result["n_4bit_weights"] += weight_count
        else:
            result["mixed_packed_bytes"] += weight_count
            result["n_8bit_weights"] += weight_count
        result["mixed_packed_bytes"] += metadata_bytes + bias_bytes
        result["bias_params"] += bias_count
    return result

# ============================================================
# REAL QUANTIZATION HELPERS (FIXED)
# ============================================================
def _checked_qparams(fake_quant):
    """Return the exact qparams used by a calibrated FakeQuantize module."""
    scale = fake_quant.scale.detach().clone().to(torch.float32)
    zero_point = fake_quant.zero_point.detach().clone().to(torch.int32)
    if scale.numel() == 0 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("FakeQuantize contains invalid or uncalibrated scales")
    return scale, zero_point


def _fake_quant_real(x, scale, zero_point, quant_min, quant_max):
    """Use PyTorch's exact affine rounding semantics for activations."""
    return torch.fake_quantize_per_tensor_affine(
        x,
        float(scale),
        int(zero_point),
        quant_min,
        quant_max,
    )


def _reshape_per_channel_qparam(qparam, weight, ch_axis):
    shape = [1] * weight.ndim
    shape[ch_axis] = -1
    return qparam.reshape(shape)


def _quantize_weight_int8(weight, scale, zero_point, quant_min, quant_max, ch_axis=0):
    """Store signed INT4/INT8 values in an int8 tensor for reference execution."""
    scale_view = _reshape_per_channel_qparam(scale, weight, ch_axis)
    zp_view = _reshape_per_channel_qparam(zero_point, weight, ch_axis)
    w_q = torch.clamp(torch.round(weight.detach() / scale_view) + zp_view,
                      quant_min, quant_max)
    return w_q.to(torch.int8)

# ============================================================
# DATA EXTRACTION
# ============================================================
class BeatExtractor:
    def __init__(self, dataset_path, window_size=SIGNAL_LENGTH):
        self.dataset_path = dataset_path
        self.window_size = window_size
        self.mapping = {
            'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
            'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
            'V': 'V', 'E': 'V',
            'F': 'F',
            'Q': 'Q', '/': 'Q', 'f': 'Q'
        }

    def extract(self):
        all_beats, all_labels = [], []
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(
                f"MIT-BIH dataset directory not found: {self.dataset_path}. "
                "Set MITBIH_DATASET_PATH to the directory containing .hea/.dat/.atr files."
            )
        files = os.listdir(self.dataset_path)
        record_sets = {
            suffix: {
                os.path.splitext(name)[0]
                for name in files
                if name.lower().endswith(suffix)
            }
            for suffix in ('.hea', '.dat', '.atr')
        }
        records = sorted(set.intersection(*record_sets.values()))
        if not records:
            raise RuntimeError(
                f"No complete MIT-BIH records found in {self.dataset_path}; "
                "each record needs matching .hea, .dat and .atr files"
            )
        incomplete = set.union(*record_sets.values()) - set(records)
        if incomplete:
            raise RuntimeError(
                "Incomplete MIT-BIH records: " + ", ".join(sorted(incomplete)) +
                ". Download matching .hea, .dat and .atr files before evaluation."
            )
        print(f"Extracting beats from {len(records)} records...")
        for name in tqdm(records):
            try:
                rec = wfdb.rdrecord(os.path.join(self.dataset_path, name))
                ann = wfdb.rdann(os.path.join(self.dataset_path, name), 'atr')
                sig, peaks = rec.p_signal[:, 0], ann.sample
                half = self.window_size // 2
                for i in range(1, len(peaks) - 1):
                    label = self.mapping.get(ann.symbol[i])
                    start = peaks[i] - half
                    if label and start > 0 and start + self.window_size < len(sig):
                        all_beats.append(sig[start:start + self.window_size])
                        all_labels.append(label)
            except Exception as exc:
                raise RuntimeError(f"Failed to extract MIT-BIH record '{name}'") from exc
        return np.array(all_beats), np.array(all_labels)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class DSCConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class InceptionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert channels % 4 == 0
        branch_ch = channels // 4
        self.bottleneck = DSCConv1d(channels, branch_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.branch_d1 = DSCConv1d(branch_ch, branch_ch, kernel_size=3, dilation=1, padding=1, stride=1)
        self.branch_d2 = DSCConv1d(branch_ch, branch_ch, kernel_size=3, dilation=2, padding=2, stride=1)
        self.branch_d3 = DSCConv1d(branch_ch, branch_ch, kernel_size=3, dilation=3, padding=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        return self.relu(torch.cat([b, self.branch_d1(b), self.branch_d2(b), self.branch_d3(b)], dim=1))

class PaperInceptionCNN(nn.Module):
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()
        self.conv1 = DSCConv1d(in_channels, 8, kernel_size=3, dilation=3, padding=3, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionBlock(8)
        self.conv2 = DSCConv1d(8, 16, kernel_size=3, dilation=2, padding=2, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception2 = InceptionBlock(16)
        self.conv3 = DSCConv1d(16, 32, kernel_size=3, dilation=1, padding=1, stride=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionBlock(32)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.inception1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.inception2(x)
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.inception3(x)
        return self.fc(self.global_pool(x).squeeze(-1))

# ============================================================
# FAKEQUANT WRAPPERS (FIXED BIAS HANDLING)
# ============================================================
class FakeQuantConv1d(nn.Module):
    """Conv1d with fake quantization for weights and activations."""
    def __init__(self, conv, weight_bits=8, act_bits=8):
        super().__init__()
        self.conv = conv
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        # Weight FakeQuantize (per-channel, symmetric)
        w_qmin = -(2 ** (weight_bits - 1))
        w_qmax = 2 ** (weight_bits - 1) - 1
        self.weight_fq = FakeQuantize(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=w_qmin, quant_max=w_qmax,
            dtype=torch.qint8, ch_axis=0, averaging_constant=0.01,
        )
        # Activation FakeQuantize (per-tensor, asymmetric)
        a_qmax = 2 ** act_bits - 1
        self.act_fq = FakeQuantize(
            observer=MovingAverageMinMaxObserver,
            quant_min=0, quant_max=a_qmax,
            dtype=torch.quint8, averaging_constant=0.01,
        )

    def forward(self, x):
        x = self.act_fq(x)
        w = self.weight_fq(self.conv.weight)
        # FIX: Xuly bias=None
        bias = self.conv.bias if self.conv.bias is not None else None
        return F.conv1d(x, w, bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

class FakeQuantLinear(nn.Module):
    """Linear layer with fake quantization for weights and activations."""
    def __init__(self, linear, weight_bits=8, act_bits=8):
        super().__init__()
        self.linear = linear
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        w_qmin = -(2 ** (weight_bits - 1))
        w_qmax = 2 ** (weight_bits - 1) - 1
        self.weight_fq = FakeQuantize(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=w_qmin, quant_max=w_qmax,
            dtype=torch.qint8, ch_axis=0, averaging_constant=0.01,
        )
        a_qmax = 2 ** act_bits - 1
        self.act_fq = FakeQuantize(
            observer=MovingAverageMinMaxObserver,
            quant_min=0, quant_max=a_qmax,
            dtype=torch.quint8, averaging_constant=0.01,
        )

    def forward(self, x):
        x = self.act_fq(x)
        w = self.weight_fq(self.linear.weight)
        # FIX: Xuly bias=None
        bias = self.linear.bias if self.linear.bias is not None else None
        return F.linear(x, w, bias)

# ============================================================
# REAL QUANT WRAPPERS (FIXED BIAS HANDLING)
# ============================================================
class RealQuantConv1d(nn.Module):
    """Reference Conv1d with real INT4/INT8 integer weight storage."""
    def __init__(self, fq_module):
        super().__init__()
        conv = fq_module.conv
        self.weight_bits = fq_module.weight_bits
        self.quant_min = fq_module.weight_fq.quant_min
        self.quant_max = fq_module.weight_fq.quant_max
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        weight_scale, weight_zero_point = _checked_qparams(fq_module.weight_fq)
        act_scale, act_zero_point = _checked_qparams(fq_module.act_fq)
        scale_view = weight_scale.reshape(-1, 1, 1)
        zp_view = weight_zero_point.reshape(-1, 1, 1)
        weight_int = torch.clamp(
            torch.round(conv.weight.detach() / scale_view) + zp_view,
            self.quant_min,
            self.quant_max,
        ).to(torch.int8)

        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('weight_zero_point', weight_zero_point)
        self.register_buffer('weight_int8', weight_int)
        self.register_buffer('act_scale', act_scale.reshape(()))
        self.register_buffer('act_zero_point', act_zero_point.reshape(()))
        self.register_buffer(
            'bias', conv.bias.detach().clone() if conv.bias is not None else None
        )

    def forward(self, x):
        x_q = _fake_quant_real(x, self.act_scale, self.act_zero_point, 0, 255)
        scale_view = self.weight_scale.reshape(-1, 1, 1)
        zp_view = self.weight_zero_point.reshape(-1, 1, 1)
        w_deq = (self.weight_int8.float() - zp_view.float()) * scale_view
        return F.conv1d(x_q, w_deq, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class RealQuantLinear(nn.Module):
    """Reference Linear layer with real INT4/INT8 integer weight storage."""
    def __init__(self, fq_module):
        super().__init__()
        linear = fq_module.linear
        self.weight_bits = fq_module.weight_bits
        self.quant_min = fq_module.weight_fq.quant_min
        self.quant_max = fq_module.weight_fq.quant_max

        weight_scale, weight_zero_point = _checked_qparams(fq_module.weight_fq)
        act_scale, act_zero_point = _checked_qparams(fq_module.act_fq)
        scale_view = weight_scale.reshape(-1, 1)
        zp_view = weight_zero_point.reshape(-1, 1)
        weight_int = torch.clamp(
            torch.round(linear.weight.detach() / scale_view) + zp_view,
            self.quant_min,
            self.quant_max,
        ).to(torch.int8)

        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('weight_zero_point', weight_zero_point)
        self.register_buffer('weight_int8', weight_int)
        self.register_buffer('act_scale', act_scale.reshape(()))
        self.register_buffer('act_zero_point', act_zero_point.reshape(()))
        self.register_buffer(
            'bias', linear.bias.detach().clone() if linear.bias is not None else None
        )

    def forward(self, x):
        x_q = _fake_quant_real(x, self.act_scale, self.act_zero_point, 0, 255)
        scale_view = self.weight_scale.reshape(-1, 1)
        zp_view = self.weight_zero_point.reshape(-1, 1)
        w_deq = (self.weight_int8.float() - zp_view.float()) * scale_view
        return F.linear(x_q, w_deq, self.bias)

# ============================================================
# MODEL WRAPPING (FIXED 4-BIT LAYER DETECTION)
# ============================================================
def replace_module(model, name, new_module):
    """Thay the mot submodule theo ten."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)

def wrap_model_with_fq(model, four_bit_layers):
    """Replace Conv1d and Linear layers with mixed-precision fake-quant modules."""
    clean_four_bit = {normalize_layer_name(ln) for ln in four_bit_layers}
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            bits = 4 if name in clean_four_bit else 8
            fq = FakeQuantConv1d(module, weight_bits=bits)
            replacements.append((name, fq, bits))
        elif isinstance(module, nn.Linear):
            bits = 4 if name in clean_four_bit else 8
            fq = FakeQuantLinear(module, weight_bits=bits)
            replacements.append((name, fq, bits))

    four_bit_count = sum(1 for _, _, b in replacements if b == 4)
    eight_bit_count = sum(1 for _, _, b in replacements if b == 8)

    for name, new_module, _ in replacements:
        replace_module(model, name, new_module)

    print(f"Wrapped {len(replacements)} layers: {four_bit_count} x 4-bit, {eight_bit_count} x 8-bit")
    return replacements

def convert_to_real_quant(model):
    """Convert fake-quant layers using their exact bit-width and qparams."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantConv1d):
            replacements.append((name, RealQuantConv1d(module)))
        elif isinstance(module, FakeQuantLinear):
            replacements.append((name, RealQuantLinear(module)))

    for name, new_module in replacements:
        replace_module(model, name, new_module)
    print(f"[CONVERT] Converted {len(replacements)} FakeQuant -> RealQuant layers")
    return replacements


def assert_real_quant_ranges(model):
    """Fail fast if a converted layer stores values outside its bit range."""
    checked = 0
    for name, module in model.named_modules():
        if not isinstance(module, (RealQuantConv1d, RealQuantLinear)):
            continue
        checked += 1
        value_min = int(module.weight_int8.min())
        value_max = int(module.weight_int8.max())
        if value_min < module.quant_min or value_max > module.quant_max:
            raise AssertionError(
                f"{name}: integer weight range [{value_min}, {value_max}] is outside "
                f"[{module.quant_min}, {module.quant_max}]"
            )
    if checked == 0:
        raise AssertionError("No real-quant layers were found")


def reset_and_enable_observers(model):
    """Start calibration from training data instead of checkpoint history."""
    for module in model.modules():
        if isinstance(module, (FakeQuantConv1d, FakeQuantLinear)):
            module.weight_fq.activation_post_process.reset_min_max_vals()
            module.act_fq.activation_post_process.reset_min_max_vals()
    model.apply(enable_observer)

# ============================================================
# DATASET
# ============================================================
class ECGBeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), QAT_GRAD_CLIP)
        optimizer.step()
        running_loss += loss.item()
        correct += (output.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / len(loader), correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    # eval() alone does not disable FakeQuantize observers.
    model.apply(disable_observer)
    model.eval()
    running_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        running_loss += loss.item()
        correct += (output.argmax(1) == y).sum().item()
        total += y.size(0)
        all_preds.extend(output.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return running_loss / len(loader), correct / total, all_labels, all_preds

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)

    print("=" * 70)
    print("MIXED PRECISION QAT OPTIMIZED (8-bit + 4-bit) - PaperInceptionCNN")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("[WARNING] Bundled checkpoints use a historical random beat-level split; ")
    print("          do not claim unseen-patient generalization without retraining.")

    # 1. Load QAT report
    print(f"\n[LOAD] Loading INT8 sensitivity data: {QAT_REPORT_PATH}")
    with open(QAT_REPORT_PATH, "r") as f:
        qat_report = json.load(f)

    # 2. Phan tich sensitivity -> xac dinh layer 4-bit / 8-bit
    sensitivity = qat_report["per_layer_sensitivity_leave_one_out"]
    sorted_sensitivity = sorted(
        sensitivity.items(), key=lambda x: x[1]["delta_acc_vs_full_int8"]
    )

    n_total = len(sorted_sensitivity)
    n_4bit = max(1, int(n_total * SENSITIVITY_RATIO_4BIT))

    four_bit_layers = [name for name, _ in sorted_sensitivity[:n_4bit]]
    eight_bit_layers = [name for name, _ in sorted_sensitivity[n_4bit:]]

    print(f"\n[DATA] Phan bo precision ({SENSITIVITY_RATIO_4BIT*100:.0f}% bottom -> 4-bit):")
    print(f"   4-bit ({len(four_bit_layers)}): {four_bit_layers}")
    print(f"   8-bit ({len(eight_bit_layers)}): {eight_bit_layers}")

    # 3. Extract data
    extractor = BeatExtractor(DATASET_PATH)
    X, y = extractor.extract()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()
    num_classes = len(class_names)

    print(f"\n[DATA] Total raw samples: {len(X):,}")
    for i, name in enumerate(class_names):
        print(f"   {name}: {np.sum(y_encoded == i):,}")

    # 4. Normalization & split
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    X_reshaped = X_norm.reshape(-1, 1, SIGNAL_LENGTH).astype(np.float32)

    X_trainfull, X_test, y_trainfull, y_test = train_test_split(
        X_reshaped, y_encoded, test_size=TEST_RATIO, random_state=SEED, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainfull, y_trainfull, test_size=VAL_RATIO, random_state=SEED, stratify=y_trainfull
    )

    train_loader = DataLoader(ECGBeatDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGBeatDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ECGBeatDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[DIR] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 5. Load FP32 baseline
    model_fp32 = PaperInceptionCNN(num_classes=num_classes)
    model_fp32.load_state_dict(torch.load(
        BASELINE_MODEL_PATH, map_location="cpu", weights_only=True
    ))

    weights = torch.tensor(CLASS_WEIGHTS[:num_classes], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    _, fp32_acc, _, _ = validate(model_fp32.to(DEVICE), test_loader, criterion, DEVICE)
    print(f"\n[PERF] FP32 baseline accuracy: {fp32_acc:.4f}")

    total_params = get_model_size(model_fp32)
    size_estimate = estimate_deployment_sizes(model_fp32, four_bit_layers)

    # 6. Wrap the model with mixed-precision fake quantization.
    print("\n[SETUP] Applying mixed-precision fake quantization...")
    wrap_model_with_fq(model_fp32, four_bit_layers)
    model = model_fp32.to(DEVICE)

    # 7. Load best QAT model (bo qua training)
    best_qat_path = QAT_MODEL_PATH
    print(f"\n[LOAD] Loading the best QAT model: {best_qat_path}")
    model.load_state_dict(torch.load(
        best_qat_path, map_location="cpu", weights_only=True
    ))
    model = model.to(DEVICE)
    print(f"[OK] Loaded best QAT model")

    # Calibrate only with training data. This must happen before test evaluation.
    print("\n[SETUP] Calibrating real quantization...")
    model.eval()
    reset_and_enable_observers(model)
    cal_samples = min(2048, len(X_train))
    cal_loader = DataLoader(ECGBeatDataset(X_train[:cal_samples], y_train[:cal_samples]),
                            batch_size=64, shuffle=False)
    with torch.no_grad():
        for x_cal, _ in cal_loader:
            model(x_cal.to(DEVICE))
    model.apply(disable_observer)
    print(f"[OK] Calibration hoan tat ({cal_samples} samples)")

    # Evaluate fake-quant only after observers are frozen, preventing test leakage.
    _, fq_acc, y_true_fq, y_pred_fq = validate(model, test_loader, criterion, DEVICE)
    print(f"\n[PERF] Mixed QAT (fake-quant) accuracy: {fq_acc:.4f}")

    print("\n[CONVERT] Converting to a real quantized reference model...")
    convert_to_real_quant(model)
    assert_real_quant_ranges(model)
    model = model.to(DEVICE)
    _, real_quant_acc, y_true_rq, y_pred_rq = validate(model, test_loader, criterion, DEVICE)
    print(f"\n[PERF] Mixed QAT (real-quant) accuracy: {real_quant_acc:.4f}")

    # 9. Calculate metrics.
    def compute_metrics(y_true, y_pred):
        if len(y_true) == 0:
            return None
        p = precision_score(y_true, y_pred, average=None, zero_division=0)
        r = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        return {
            "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
                          for i, name in enumerate(class_names)},
            "macro": {
                "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            },
        }

    fq_metrics = compute_metrics(y_true_fq, y_pred_fq)
    rq_metrics = compute_metrics(y_true_rq, y_pred_rq)

    # Compare raw deployment estimates with the same accounting method.
    fp32_size_bytes = size_estimate["fp32_raw_bytes"]
    full_int8_size_bytes = size_estimate["full_int8_packed_bytes"]
    mixed_size_bytes = size_estimate["mixed_packed_bytes"]
    compression_ratio = fp32_size_bytes / mixed_size_bytes if mixed_size_bytes else 0
    reference_storage_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.buffers()
    )

    # 11. Save JSON report
    full_int8_fakequant_acc = qat_report["accuracy"]["after_qat_fakequant"]
    full_int8_realquant_acc = qat_report["accuracy"]["after_int8_convert"]

    report = {
        "model": "PaperInceptionCNN_DSC_noBN_MixedPrecision",
        "group_id": GROUP_ID,
        "total_params": total_params,
        "evaluation_protocol": {
            "split": "stratified_random_beat",
            "patient_disjoint": False,
            "warning": "Bundled checkpoints require patient-disjoint retraining for unseen-patient claims.",
        },
        "mixed_precision_strategy": {
            "method": "sensitivity_based_leave_one_out",
            "four_bit_ratio": SENSITIVITY_RATIO_4BIT,
            "four_bit_layers": four_bit_layers,
            "eight_bit_layers": eight_bit_layers,
            "activation_bits": 8,
            "n_4bit_weights": size_estimate["n_4bit_weights"],
            "n_8bit_weights": size_estimate["n_8bit_weights"],
            "fp32_bias_params": size_estimate["bias_params"],
        },
        "accuracy": {
            "fp32_baseline": fp32_acc,
            "full_int8_fakequant_baseline": full_int8_fakequant_acc,
            "full_int8_realquant_baseline": full_int8_realquant_acc,
            "mixed_precision_fakequant": fq_acc,
            "mixed_precision_realquant": real_quant_acc,
        },
        "delta_vs_full_int8": {
            "mixed_fakequant_vs_full_int8_fakequant": fq_acc - full_int8_fakequant_acc,
            "mixed_realquant_vs_full_int8_realquant": real_quant_acc - full_int8_realquant_acc,
            "realquant_vs_fakequant": real_quant_acc - fq_acc,
        },
        "model_size": {
            "fp32_raw_parameter_bytes": fp32_size_bytes,
            "full_int8_packed_estimate_bytes": full_int8_size_bytes,
            "mixed_int4_int8_packed_estimate_bytes": mixed_size_bytes,
            "python_reference_unpacked_buffer_bytes": reference_storage_bytes,
            "compression_ratio_vs_fp32": compression_ratio,
            "note": "Packed estimates include per-channel weight qparams, per-layer activation qparams, and FP32 bias.",
        },
        "metrics": {"fakequant": fq_metrics, "realquant": rq_metrics},
        "checkpoint_evaluation": {
            "trained_in_this_run": False,
            "source_qat_epochs": qat_report.get("qat_epochs"),
            "source_qat_lr": qat_report.get("qat_lr"),
            "calibration_samples": cal_samples,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
    }

    reports_dir = RESULTS_DIR
    os.makedirs(reports_dir, exist_ok=True)
    report_json_path = os.path.join(reports_dir, f"{GROUP_ID}_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 12. Save TXT report
    report_txt_path = os.path.join(reports_dir, f"{GROUP_ID}_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BAO CAO MIXED PRECISION QAT OPTIMIZED (8-bit + 4-bit)\n")
        f.write("PaperInceptionCNN with DSC (no BatchNorm)\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- Chien luoc Mixed Precision ---\n")
        f.write(f"Phuong phap: Sensitivity-based (leave-one-out analysis)\n")
        f.write(f"INT4 ratio: {SENSITIVITY_RATIO_4BIT*100:.0f}% least-sensitive layers\n")
        f.write(f"Activation: luon 8-bit\n\n")

        f.write(f"Layer 4-bit ({len(four_bit_layers)}):\n")
        for name in four_bit_layers:
            d = sensitivity[name]["delta_acc_vs_full_int8"]
            f.write(f"  {name}: delta_acc = {d:+.6f}\n")
        f.write(f"\nLayer 8-bit ({len(eight_bit_layers)}):\n")
        for name in eight_bit_layers:
            d = sensitivity[name]["delta_acc_vs_full_int8"]
            f.write(f"  {name}: delta_acc = {d:+.6f}\n")
        f.write("\n")

        f.write("--- Accuracy ---\n")
        f.write(f'{"Giai doan":<42} {"Acc":<10}\n')
        f.write("-" * 55 + "\n")
        f.write(f'{"FP32 baseline":<42} {fp32_acc:.4f}\n')
        f.write(f'{"Full INT8 fake-quant baseline":<42} {full_int8_fakequant_acc:.4f}\n')
        f.write(f'{"Full INT8 real-quant baseline":<42} {full_int8_realquant_acc:.4f}\n')
        f.write(f'{"Mixed Precision QAT (fake-quant)":<42} {fq_acc:.4f}\n')
        f.write(f'{"Mixed Precision QAT (real-quant)":<42} {real_quant_acc:.4f}\n')
        f.write("\n")

        f.write("--- Comparison with full INT8 ---\n")
        fq_delta = fq_acc - full_int8_fakequant_acc
        rq_delta = real_quant_acc - full_int8_realquant_acc
        f.write(f"  Fake-quant vs Full INT8 fake-quant: {fq_delta:+.4f}\n")
        f.write(f"  Real-quant vs Full INT8 real-quant: {rq_delta:+.4f}\n")
        f.write(f"  Real-quant vs Fake-quant: {real_quant_acc - fq_acc:+.4f}\n")
        f.write("\n")

        f.write("--- Kich thuoc model ---\n")
        f.write(f"  FP32 raw params:           {fp32_size_bytes:>8,} bytes\n")
        f.write(f"  Full INT8 packed estimate: {full_int8_size_bytes:>8,} bytes\n")
        f.write(f"  Mixed packed estimate:     {mixed_size_bytes:>8,} bytes\n")
        f.write(f"  Python reference buffers:  {reference_storage_bytes:>8,} bytes\n")
        f.write(f"  He so nen vs FP32:         {compression_ratio:.4f}x\n")
        f.write(f"  4-bit weights:      {size_estimate['n_4bit_weights']:,}\n")
        f.write(f"  8-bit weights:      {size_estimate['n_8bit_weights']:,}\n")
        f.write(f"  Tong params:        {total_params:,}\n")
        f.write("  Note: Bao gom weight qparams per-channel, activation qparams va FP32 bias.\n")
        f.write("\n")

        if fq_metrics:
            f.write("--- Chi tiet theo lop -- Fake-Quant ---\n")
            f.write("{:<12} {:<12} {:<12} {:<12}\n".format("Class", "Precision", "Recall", "F1"))
            f.write("-" * 50 + "\n")
            for name in class_names:
                m = fq_metrics["per_class"][name]
                f.write("{:<12} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                    name, m["precision"], m["recall"], m["f1"]))
            f.write("{:<12} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                "Macro", fq_metrics["macro"]["precision"],
                fq_metrics["macro"]["recall"], fq_metrics["macro"]["f1"]))
            f.write("\n")

        if rq_metrics:
            f.write("--- Chi tiet theo lop -- Real-Quant ---\n")
            f.write("{:<12} {:<12} {:<12} {:<12}\n".format("Class", "Precision", "Recall", "F1"))
            f.write("-" * 50 + "\n")
            for name in class_names:
                m = rq_metrics["per_class"][name]
                f.write("{:<12} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                    name, m["precision"], m["recall"], m["f1"]))
            f.write("{:<12} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                "Macro", rq_metrics["macro"]["precision"],
                rq_metrics["macro"]["recall"], rq_metrics["macro"]["f1"]))
            f.write("\n")

        f.write("--- Cau hinh danh gia checkpoint ---\n")
        f.write("  Trained in this run: No\n")
        f.write(f"  Source QAT epochs: {qat_report.get('qat_epochs')}, LR: {qat_report.get('qat_lr')}\n")
        f.write(f"  Batch size: {BATCH_SIZE}, Seed: {SEED}\n")
        f.write(f"  Calibration samples: {cal_samples}\n")

    # 13. Final summary
    print("\n" + "=" * 70)
    print("KET QUA MIXED PRECISION QAT OPTIMIZED")
    print("=" * 70)
    print(f"  FP32 baseline:            {fp32_acc:.4f}")
    print(f"  Full INT8 fake-quant:     {full_int8_fakequant_acc:.4f}")
    print(f"  Full INT8 real-quant:     {full_int8_realquant_acc:.4f}")
    print(f"  Mixed QAT (fake-quant):   {fq_acc:.4f}  ({fq_acc - full_int8_fakequant_acc:+.4f} vs matching INT8)")
    print(f"  Mixed QAT (real-quant):   {real_quant_acc:.4f}  ({real_quant_acc - full_int8_realquant_acc:+.4f} vs matching INT8)")
    print(f"  4-bit layers: {len(four_bit_layers)} | 8-bit layers: {len(eight_bit_layers)}")
    print(f"  FP32 raw: {fp32_size_bytes:,} bytes | Mixed packed estimate: {mixed_size_bytes:,} bytes ({compression_ratio:.2f}x smaller)")
    print(f"  Calibration: {cal_samples} samples")
    print(f"  Saved: {report_json_path}, {report_txt_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

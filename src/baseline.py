"""Section-level LSTM training and inference on CPU or CUDA. See docs/lstm-baseline.md."""

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import warnings

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

if __package__:
    from .model import SectionLSTM
else:
    from model import SectionLSTM

MIN_FEATURE_STD = 1e-6
AUGMENTATION_SEED_OFFSET = 1_000_003
MIN_AUGMENTATION_APPLY_RATE = 0.1


def validate_augmentation(config, schema):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) != {"probability", "features"}:
        raise ValueError("Augmentation requires probability and features")
    probability = config["probability"]
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Augmentation probability must be between 0 and 1")
    continuous_features = schema.get("continuous_features", [])
    if not isinstance(continuous_features, list) or any(name not in schema["feature_names"] for name in continuous_features):
        raise ValueError("Schema continuous_features must list known feature names")
    features = config["features"]
    if not isinstance(features, dict) or not features:
        raise ValueError("Augmentation features must be a nonempty object")
    required = {"bias_max", "noise_max", "smoothing", "min", "max", "protected_values"}
    for name, settings in features.items():
        if name not in continuous_features:
            raise ValueError(f"Augmented feature {name!r} must be declared continuous in schema")
        if not isinstance(settings, dict) or set(settings) != required:
            raise ValueError(f"Augmentation {name!r} requires {sorted(required)}")
        for key in required - {"protected_values"}:
            value = settings[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Augmentation {name!r}.{key} must be finite numeric")
        if settings["bias_max"] < 0 or settings["noise_max"] < 0:
            raise ValueError("Bias/noise maximum must be nonnegative")
        if not 0 < settings["smoothing"] <= 1 or settings["min"] >= settings["max"]:
            raise ValueError("Smoothing must be in (0, 1]; min must be smaller than max")
        thresholds = settings["protected_values"]
        if not isinstance(thresholds, list) or any(type(value) not in (int, float) or not math.isfinite(value) for value in thresholds):
            raise ValueError("Protected values must be a list of finite thresholds")
    return copy.deepcopy(config)


def validate_augmentation_bounds(features, feature_names, config):
    for name, settings in config["features"].items():
        original = features[:, feature_names.index(name)]
        if ((original < settings["min"]) | (original > settings["max"])).any():
            raise ValueError(f"Original feature {name!r} exceeds configured bounds")


def augment_features(features, feature_names, config, generator, stats=None):
    augmented = features.clone()
    validate_augmentation_bounds(features, feature_names, config)
    if torch.rand((), generator=generator).item() >= config["probability"]:
        if stats is not None:
            for counts in stats.values():
                counts["skipped"] += 1
        return augmented
    for name, settings in config["features"].items():
        counts = stats[name] if stats is not None else None
        if counts is not None:
            counts["attempted"] += 1
        index = feature_names.index(name)
        original = features[:, index]
        bias = (torch.rand((), generator=generator) * 2 - 1) * settings["bias_max"]
        innovations = (torch.rand(len(original), generator=generator) * 2 - 1) * settings["noise_max"]
        noise = torch.empty_like(original)
        noise[0] = innovations[0]
        smoothing = settings["smoothing"]
        for timestep in range(1, len(original)):
            noise[timestep] = (1 - smoothing) * noise[timestep - 1] + smoothing * innovations[timestep]
        candidate = original + bias + noise
        if not torch.isfinite(candidate).all():
            raise ValueError(f"Augmentation overflow for feature {name!r}")
        if ((candidate < settings["min"]) | (candidate > settings["max"])).any():
            if counts is not None:
                counts["rejected_bounds"] += 1
            continue
        # Keep each frame's relation to configured decision boundaries, including equality.
        crosses_boundary = any(
            ((original < threshold) != (candidate < threshold)).any().item()
            or ((original > threshold) != (candidate > threshold)).any().item()
            for threshold in settings["protected_values"]
        )
        if crosses_boundary:
            if counts is not None:
                counts["rejected_threshold"] += 1
            continue
        augmented[:, index] = candidate
        if counts is not None:
            counts["unchanged" if torch.equal(original, candidate) else "applied"] += 1
    return augmented


def load_records(path, feature_names, label_names, labeled=True, sample_rate_hz=50.0, data_bytes=None):
    records = []
    run_splits = {}
    driver_splits = {}
    if data_bytes is None:
        data_bytes = Path(path).read_bytes()
    source = data_bytes.decode("utf-8").splitlines()
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object")
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1 or type(record.get("sample_rate_hz")) not in (int, float) or record["sample_rate_hz"] != sample_rate_hz:
            raise ValueError(f"Line {line_number}: schema version/sample rate mismatch")
        if not isinstance(record.get("run_id"), str) or not record["run_id"].strip():
            raise ValueError(f"Line {line_number}: nonempty anonymous run_id required")
        if record.get("feature_names") != feature_names:
            raise ValueError(f"Line {line_number}: feature names/order mismatch")
        raw_features = record.get("features")
        if not isinstance(raw_features, list) or not raw_features or any(
            not isinstance(frame, list) or len(frame) != len(feature_names)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in frame)
            for frame in raw_features
        ):
            raise ValueError(f"Line {line_number}: expected finite numeric T x F features")
        features = torch.tensor(raw_features, dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != len(feature_names):
            raise ValueError(f"Line {line_number}: expected nonempty T x F features")
        if not torch.isfinite(features).all():
            raise ValueError(f"Line {line_number}: features must be finite")
        sample = dict(record, features=features)
        if labeled:
            if record.get("label_names") != label_names:
                raise ValueError(f"Line {line_number}: label names/order mismatch")
            raw_labels = record.get("labels")
            if not isinstance(raw_labels, list) or len(raw_labels) != len(label_names) or any(type(value) not in (int, float) or value not in (0, 1) for value in raw_labels):
                raise ValueError(f"Line {line_number}: labels must be a complete binary vector")
            labels = torch.tensor(raw_labels, dtype=torch.float32)
            if labels.shape != (len(label_names),) or not ((labels == 0) | (labels == 1)).all():
                raise ValueError(f"Line {line_number}: labels must be a binary vector")
            split = record.get("split")
            if split not in ("train", "val", "test"):
                raise ValueError(f"Line {line_number}: split must be train/val/test")
            if run_splits.setdefault(record["run_id"], split) != split:
                raise ValueError(f"Line {line_number}: run_id crosses splits")
            driver_id = record.get("driver_id")
            if driver_id is not None:
                if not isinstance(driver_id, str) or not driver_id.strip():
                    raise ValueError(f"Line {line_number}: driver_id must be anonymous nonempty string")
                if driver_splits.setdefault(driver_id, split) != split:
                    raise ValueError(f"Line {line_number}: driver_id crosses splits")
            sample["labels"] = labels
        records.append(sample)
    if not records:
        raise ValueError("No records found")
    return records


def fit_normalization(records):
    frames = torch.cat([record["features"] for record in records])
    mean = frames.mean(0)
    standard_deviation = frames.std(0, unbiased=False)
    scale = torch.where(standard_deviation < MIN_FEATURE_STD, torch.ones_like(standard_deviation), standard_deviation)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError("Nonfinite normalization statistics; check feature magnitudes")
    return mean, scale


def make_batch(records, mean, scale, augmentation=None, generator=None, augmentation_stats=None):
    sequences = []
    for record in records:
        features = record["features"]
        if augmentation is not None:
            features = augment_features(features, record["feature_names"], augmentation, generator, augmentation_stats)
        sequences.append((features - mean) / scale)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    return pad_sequence(sequences, batch_first=True), lengths


def validate_device_name(name):
    if not isinstance(name, str):
        raise ValueError("Device must be auto, cpu, cuda or cuda:<index>")
    if name in ("auto", "cpu", "cuda"):
        return name
    if name.startswith("cuda:") and name[5:].isascii() and name[5:].isdigit():
        return name
    raise ValueError("Device must be auto, cpu, cuda or cuda:<index>")


def resolve_device(name="auto"):
    validate_device_name(name)
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install a CUDA-enabled PyTorch build and check the NVIDIA driver")
    index = 0 if name == "cuda" else int(name[5:])
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} unavailable; found {torch.cuda.device_count()} device(s)")
    # Configure cuBLAS before the model creates CUDA tensors/handles.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return torch.device("cuda", index)


@contextmanager
def training_runtime(seed, deterministic):
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


def evaluate(model, records, mean, scale, batch_size):
    model.eval()
    device = next(model.parameters()).device
    loss_sum = 0.0
    true_positive = false_positive = false_negative = 0
    label_counts = torch.zeros(model.classifier.out_features, 4, dtype=torch.int64, device=device)
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            features, lengths = make_batch(batch, mean, scale)
            labels = torch.stack([record["labels"] for record in batch]).to(device)
            logits = model(features.to(device), lengths)
            loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item()
            predicted = logits >= 0
            actual = labels.bool()
            true_positive += (predicted & actual).sum().item()
            false_positive += (predicted & ~actual).sum().item()
            false_negative += (~predicted & actual).sum().item()
            label_counts[:, 0] += actual.sum(0)
            label_counts[:, 1] += (predicted & actual).sum(0)
            label_counts[:, 2] += (predicted & ~actual).sum(0)
            label_counts[:, 3] += (~predicted & actual).sum(0)
    denominator = 2 * true_positive + false_positive + false_negative
    return {"bce": loss_sum / (len(records) * model.classifier.out_features),
            "micro_f1": 2 * true_positive / denominator if denominator else 0.0,
            "samples": len(records),
            "per_label": [{"positive_count": count, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
                          for count, tp, fp, fn in label_counts.tolist()]}


class LSTMBaseline:
    """Owns training, normalization, checkpoint loading and inference on one device."""

    def __init__(self, device="auto"):
        self._device = resolve_device(device)
        self._model = None
        self._checkpoint = None

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    def fit(self, args):
        """Train a fresh model with resolved settings on the constructor's device and save its best weights."""
        for key in ("epochs", "batch_size", "hidden_size", "num_layers"):
            value = getattr(args, key, 1) if key == "num_layers" else getattr(args, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(args.learning_rate) not in (int, float) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        deterministic = getattr(args, "deterministic", True)
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be true or false")
        with training_runtime(args.seed, deterministic):
            return self._fit(args)

    def _fit(self, args):
        num_layers = getattr(args, "num_layers", 1)
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        sample_rate = schema.get("sample_rate_hz")
        if type(schema.get("schema_version")) is not int or schema["schema_version"] != 1 or isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)) or not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Schema requires version 1 and finite positive sample_rate_hz")
        for key in ("feature_names", "label_names"):
            names = schema[key]
            if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError(f"{key} must contain unique nonempty names")
        augmentation = validate_augmentation(getattr(args, "augmentation", None), schema)
        augmentation_seed = getattr(args, "augmentation_seed", None)
        if augmentation_seed is None:
            augmentation_seed = (args.seed + AUGMENTATION_SEED_OFFSET) % (1 << 63)
        augmentation_generator = torch.Generator().manual_seed(augmentation_seed)
        augmentation_stats = {name: {key: 0 for key in ("attempted", "applied", "unchanged", "skipped", "rejected_bounds", "rejected_threshold")}
                              for name in augmentation["features"]} if augmentation is not None else {}
        data_bytes = Path(args.data).read_bytes()
        data_sha256 = hashlib.sha256(data_bytes).hexdigest()
        records = load_records(args.data, schema["feature_names"], schema["label_names"], sample_rate_hz=sample_rate, data_bytes=data_bytes)
        splits = {split: [record for record in records if record["split"] == split] for split in ("train", "val", "test")}
        if not splits["train"] or not splits["val"]:
            raise ValueError("Nonempty train and val splits required")
        if not splits["test"]:
            warnings.warn("No held-out test split; validation metrics were used for model selection")
        if augmentation is not None:
            for record in splits["train"]:
                validate_augmentation_bounds(record["features"], schema["feature_names"], augmentation)
        mean, scale = fit_normalization(splits["train"])
        prevalence = torch.stack([record["labels"] for record in splits["train"]]).mean(0)
        for name, rate in zip(schema["label_names"], prevalence.tolist()):
            if rate in (0.0, 1.0):
                warnings.warn(f"Training label {name!r} has only one class; discrimination cannot be learned")
        model = SectionLSTM(len(schema["feature_names"]), len(schema["label_names"]), args.hidden_size, num_layers=num_layers).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_loss = math.inf
        best_state = None
        best_epoch = None
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(splits["train"])).tolist()
            for start in range(0, len(order), args.batch_size):
                batch = [splits["train"][index] for index in order[start:start + args.batch_size]]
                features, lengths = make_batch(batch, mean, scale, augmentation, augmentation_generator, augmentation_stats)
                features = features.to(self.device)
                labels = torch.stack([record["labels"] for record in batch]).to(self.device)
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(features, lengths), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; check feature magnitudes")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics = evaluate(model, splits["val"], mean, scale, args.batch_size)
            if not math.isfinite(metrics["bce"]):
                raise ValueError("Nonfinite validation loss")
            if metrics["bce"] < best_loss:
                best_loss, best_state, best_epoch = metrics["bce"], {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, epoch + 1
        model.load_state_dict(best_state)
        metrics = {split: evaluate(model, samples, mean, scale, args.batch_size) for split, samples in splits.items() if samples}
        constant_model = SectionLSTM(len(schema["feature_names"]), len(schema["label_names"]), args.hidden_size, num_layers=num_layers).to(self.device)
        with torch.no_grad():
            constant_model.classifier.weight.zero_()
            constant_model.classifier.bias.copy_(torch.logit(prevalence.clamp(1e-6, 1 - 1e-6)).to(self.device))
        reference_metrics = {split: evaluate(constant_model, samples, mean, scale, args.batch_size) for split, samples in splits.items() if samples}
        for name, counts in augmentation_stats.items():
            if counts["attempted"] and counts["applied"] / counts["attempted"] < MIN_AUGMENTATION_APPLY_RATE:
                warnings.warn(f"Augmentation {name!r} applied to less than 10% of attempts; inspect bounds/thresholds and strength")
        checkpoint = {"format_version": 1, "schema": schema, "hidden_size": args.hidden_size, "num_layers": num_layers,
                      "mean": mean, "scale": scale, "state_dict": best_state,
                      "seed": args.seed, "best_epoch": best_epoch, "metrics": metrics,
                      "prevalence_baseline": reference_metrics,
                      "augmentation": augmentation,
                      "augmentation_seed": augmentation_seed, "augmentation_stats": augmentation_stats,
                      "training_args": {key: value for key, value in vars(args).items() if key not in ("data", "schema", "output", "config")},
                      "device": str(self.device), "cuda_version": torch.version.cuda,
                      "torch_version": str(torch.__version__),
                      "python_version": sys.version, "data_sha256": data_sha256}
        checkpoint["training_args"]["device"] = str(self.device)
        checkpoint["training_args"]["num_layers"] = num_layers
        checkpoint["training_args"]["deterministic"] = getattr(args, "deterministic", True)
        self._model = model
        self._checkpoint = checkpoint
        self.save(args.output)
        return {"lstm": metrics, "prevalence_baseline": reference_metrics, "label_names": schema["label_names"],
                "augmentation_stats": augmentation_stats, "device": str(self.device)}

    def save(self, path):
        self._require_model()
        checkpoint = dict(self._checkpoint)
        checkpoint["state_dict"] = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        checkpoint["mean"] = checkpoint["mean"].cpu()
        checkpoint["scale"] = checkpoint["scale"].cpu()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device="auto"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        baseline = cls(device)
        schema = checkpoint["schema"]
        baseline._model = SectionLSTM(len(schema["feature_names"]), len(schema["label_names"]), checkpoint["hidden_size"],
                                     num_layers=checkpoint.get("num_layers", 1)).to(baseline.device)
        baseline.model.load_state_dict(checkpoint["state_dict"])
        baseline.model.eval()
        baseline._checkpoint = checkpoint
        return baseline

    def evaluate(self, data_path, batch_size=32):
        self._require_model()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        schema = self._checkpoint["schema"]
        records = load_records(data_path, schema["feature_names"], schema["label_names"], sample_rate_hz=schema["sample_rate_hz"])
        return evaluate(self.model, records, self._checkpoint["mean"], self._checkpoint["scale"], batch_size)

    def predict(self, data_path):
        self._require_model()
        schema = self._checkpoint["schema"]
        records = load_records(data_path, schema["feature_names"], schema["label_names"], labeled=False, sample_rate_hz=schema["sample_rate_hz"])
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for record in records:
                features, lengths = make_batch([record], self._checkpoint["mean"], self._checkpoint["scale"])
                probabilities = self.model(features.to(self.device), lengths).sigmoid()[0].cpu().tolist()
                predictions.append({"run_id": record["run_id"], "section_id": record.get("section_id"),
                                    "schema_version": schema["schema_version"], "model_format_version": self._checkpoint["format_version"],
                                    "probabilities": dict(zip(schema["label_names"], probabilities))})
        return predictions

    def _require_model(self):
        if self.model is None or self._checkpoint is None:
            raise ValueError("Fit or load a model before evaluation, prediction or saving")


def train(args):
    baseline = LSTMBaseline(getattr(args, "device", "auto"))
    result = baseline.fit(args)
    print(json.dumps(result))
    return baseline


def predict(args):
    baseline = LSTMBaseline.load(args.checkpoint, device=getattr(args, "device", "auto"))
    for result in baseline.predict(args.data):
        print(json.dumps(result))


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", help="JSON training settings; explicit CLI values override these")
    training.add_argument("--data")
    training.add_argument("--schema")
    training.add_argument("--output")
    training.add_argument("--augmentation", help="JSON augmentation settings; overrides config augmentation")
    training.add_argument("--epochs", type=positive_int)
    training.add_argument("--batch-size", type=positive_int)
    training.add_argument("--hidden-size", type=positive_int)
    training.add_argument("--num-layers", type=positive_int, help="Number of stacked LSTM layers (default: 1)")
    training.add_argument("--learning-rate", type=positive_float)
    training.add_argument("--seed", type=int)
    training.add_argument("--augmentation-seed", type=int, help="Independent augmentation RNG seed")
    training.add_argument("--device", help="auto, cpu, cuda or cuda:<index>")
    training.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    inference = commands.add_parser("predict")
    inference.add_argument("--data", required=True)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:<index>")
    args = parser.parse_args(argv)
    if args.command != "train":
        try:
            validate_device_name(args.device)
        except ValueError as error:
            parser.error(str(error))
        return args
    defaults = {"epochs": 20, "batch_size": 32, "hidden_size": 64, "num_layers": 1, "learning_rate": 0.001, "seed": 42,
                "data": None, "schema": None, "output": None, "augmentation": None, "augmentation_seed": None,
                "device": "auto", "deterministic": True}
    configured = {}
    try:
        if args.config:
            config_path = Path(args.config)
            configured = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(configured, dict) or set(configured) - set(defaults):
                raise ValueError("Unknown training configuration keys")
            for key in ("data", "schema", "output"):
                if key in configured:
                    if not isinstance(configured[key], str) or not configured[key].strip():
                        raise ValueError(f"{key} must be a nonempty path")
                    configured[key] = str((config_path.parent / configured[key]).resolve())
        for key, default in defaults.items():
            cli_value = getattr(args, key)
            value = cli_value if cli_value is not None else configured.get(key, default)
            if key in ("epochs", "batch_size", "hidden_size", "num_layers", "seed"):
                if type(value) is not int or (key != "seed" and value <= 0):
                    raise ValueError(f"{key} must be {'an integer' if key == 'seed' else 'a positive integer'}")
            if key == "learning_rate" and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError("learning_rate must be finite and positive")
            if key == "augmentation_seed" and value is not None and type(value) is not int:
                raise ValueError("augmentation_seed must be an integer or null")
            if key == "device":
                validate_device_name(value)
            if key == "deterministic" and type(value) is not bool:
                raise ValueError("deterministic must be true or false")
            if key in ("data", "schema", "output") and not value:
                raise ValueError(f"Supply --{key} or {key} in --config")
            if key == "augmentation" and cli_value is not None:
                value = json.loads(Path(cli_value).read_text(encoding="utf-8"))
            setattr(args, key, value)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()

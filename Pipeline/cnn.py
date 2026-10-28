"""MobileNetV2 transfer learning for RealWaste.

Speed-ups compared with training the whole network on every experiment:
  1. FROZEN-FEATURE CACHE: while the backbone is frozen its output never changes, so features of the clean images
     plus a few augmented views are computed ONCE (a forward pass only) and cached to Data/cache. Head
     hyper-parameter search and head training then run on 1280-d vectors in seconds per epoch instead of minutes.
  2. The trained head is copied into the full model (same layer names), which is then fine-tuned end-to-end.
  3. Every trained stage is saved with a fingerprint of the data split and is re-loaded instead of re-trained.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .config import CATEGORIES, IMG_SIZE

NUM_CLASSES = len(CATEGORIES)
FEAT_DIM = 1280
HEAD_LAYERS = ("head_bn", "head_dense", "waste_class")


# ----------------------------------------------------------------------------- building blocks
def one_hot(y):
    return np.eye(NUM_CLASSES, dtype="float32")[np.asarray(y)]


def class_weights_for(y) -> dict:
    w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=np.asarray(y))
    return {i: float(x) for i, x in enumerate(w)}


def make_augmentation():
    """Flip / rotate / zoom / contrast. Only active when called with training=True."""
    return tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.15),
                                layers.RandomZoom(0.15), layers.RandomContrast(0.15)], name="augmentation")


def _head(x, units, dropout, l2):
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(l2), name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", name="waste_class")(x)


def _compile(model, lr, label_smoothing=0.05):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), metrics=["accuracy"])
    return model


def build_head_model(units=256, dropout=0.4, l2=1e-4, lr=1e-3):
    """Stand-alone head: 1280-d MobileNetV2 features -> 9 classes (trained on cached features)."""
    inp = layers.Input((FEAT_DIM,), name="features")
    return _compile(models.Model(inp, _head(inp, units, dropout, l2), name="head"), lr)


def build_cnn(units=256, dropout=0.4, l2=1e-4, lr=1e-3, head: tf.keras.Model | None = None):
    """Full model: raw 0-255 pixels -> augmentation -> [-1,1] rescale -> MobileNetV2 (frozen) -> custom head.
    Pre-processing lives INSIDE the model so the saved model accepts raw pixels (no train/serve mismatch)."""
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
    base.trainable = False
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="raw_pixels_0_255")
    x = make_augmentation()(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess")(x)
    x = base(x, training=False)                         # keep BatchNorm statistics frozen
    model = models.Model(inputs, _head(x, units, dropout, l2), name="ecosort_cnn")
    if head is not None:
        for name in HEAD_LAYERS:
            model.get_layer(name).set_weights(head.get_layer(name).get_weights())
    return _compile(model, lr)


def backbone_of(model):
    return next(l for l in model.layers if l.name.startswith("mobilenetv2"))


# ----------------------------------------------------------------------------- cached frozen features
_PRISTINE = None
def _pristine_backbone():
    global _PRISTINE
    if _PRISTINE is None:
        _PRISTINE = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
        _PRISTINE.trainable = False
    return _PRISTINE


def _make_forward(augment):
    """Graph-compiled forward pass (eager per-batch calls are several times slower on a CPU)."""
    bb, aug = _pristine_backbone(), (make_augmentation() if augment else None)

    @tf.function(reduce_retracing=True)
    def fwd(xb):
        xb = tf.cast(xb, tf.float32)
        if aug is not None:
            xb = aug(xb, training=True)
        return bb(xb / 127.5 - 1.0, training=False)
    return fwd


def extract_features(X, augment=False, batch_size=128, seed=0):
    fwd = _make_forward(augment)
    tf.random.set_seed(seed)
    return np.concatenate([fwd(tf.constant(X[i:i + batch_size])).numpy() for i in range(0, len(X), batch_size)])


def get_feature_set(data, n_aug: int = 1, force: bool = False, verbose: bool = True) -> dict:
    """Frozen-backbone features: train = clean + n_aug augmented views, val/test = clean. Cached on disk."""
    config.ensure_dirs()
    path = config.CACHE_DIR / f"feats_{data.fingerprint}_aug{n_aug}.npz"
    t0 = time.time()
    if path.exists() and not force:
        z = np.load(path); out = {k: z[k] for k in z.files}; src = "cache"
    else:
        tr = [extract_features(data.X_train)] + [extract_features(data.X_train, augment=True, seed=s + 1) for s in range(n_aug)]
        out = dict(train_X=np.concatenate(tr), train_y=np.tile(data.y_train, n_aug + 1),
                   val_X=extract_features(data.X_val), val_y=data.y_val,
                   test_X=extract_features(data.X_test), test_y=data.y_test)
        np.savez(path, **out); src = "computed"
    if verbose:
        print(f"Frozen features ({src}) in {time.time() - t0:.1f}s | train {out['train_X'].shape}, val {out['val_X'].shape}")
    return out


def train_head(feats, cfg, epochs=25, patience=3, batch_size=256, verbose=0):
    head = build_head_model(**cfg)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    h = head.fit(feats["train_X"], one_hot(feats["train_y"]), validation_data=(feats["val_X"], one_hot(feats["val_y"])),
                 epochs=epochs, batch_size=batch_size, class_weight=class_weights_for(feats["train_y"]), callbacks=[es], verbose=verbose)
    return head, h


def head_search(feats, space, epochs=25, verbose=True, use_cache=True) -> tuple[pd.DataFrame, dict]:
    """Grid search over head settings on cached features. The result table is cached on disk (keyed by the
    features and the grid), so re-running the notebook does not repeat the search."""
    config.ensure_dirs()
    key = config.fingerprint(space, epochs, list(feats["train_X"].shape), float(np.round(feats["train_X"][:64].sum(), 3)))
    path = config.CACHE_DIR / f"headsearch_{key}.json"
    if use_cache and path.exists():
        df = pd.read_json(path)
        if verbose: print("Head search loaded from cache.")
    else:
        rows = []
        for n, cfg in enumerate(space, 1):
            t0 = time.time()
            head, h = train_head(feats, cfg, epochs=epochs)
            i = int(np.argmin(h.history["val_loss"]))
            rows.append({**cfg, "val_loss": h.history["val_loss"][i], "val_acc": h.history["val_accuracy"][i],
                         "train_acc": h.history["accuracy"][i], "epochs_run": len(h.history["loss"])})
            if verbose: print(f"  [{n}/{len(space)}] {cfg} -> val_acc {rows[-1]['val_acc']:.4f} ({time.time() - t0:.0f}s)")
        df = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)
        df.to_json(path)
    df = df.sort_values("val_loss").reset_index(drop=True)
    best = {k: (int(df.loc[0, k]) if k == "units" else float(df.loc[0, k])) for k in ("units", "dropout", "l2", "lr")}
    return df, best


# ----------------------------------------------------------------------------- fine-tuning (with a trained-stage cache)
def unfreeze_top(model, n_layers):
    bb = backbone_of(model)
    bb.trainable = True
    for l in bb.layers[:-n_layers]:
        l.trainable = False
    for l in bb.layers:                                   # never un-freeze BatchNorm (small-batch stability)
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def finetune(model, data, depth=30, epochs=6, lr=1e-5, patience=3, verbose=1):
    unfreeze_top(model, depth)
    _compile(model, lr)
    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6)]
    return model.fit(data.X_train, one_hot(data.y_train), validation_data=(data.X_val, one_hot(data.y_val)),
                     batch_size=config.BATCH_SIZE, epochs=epochs, class_weight=class_weights_for(data.y_train),
                     callbacks=cbs, verbose=verbose)


def save_cnn(model, path, data, **meta):
    model.save(path)
    config.write_meta(path, data_fingerprint=data.fingerprint, **meta)


def load_cnn(path=None, data=None):
    """Load a saved model (inference only). If `data` is given, return None when the model was trained on a
    different split (stale), so the caller re-trains instead of silently evaluating on leaked images."""
    path = path or config.CNN_FINAL
    if not path.exists():
        return None
    if data is not None and config.read_meta(path).get("data_fingerprint") != data.fingerprint:
        return None
    return tf.keras.models.load_model(path, compile=False)


def finetune_or_load(head_model_cfg, head, data, depth, epochs, tag, force=False):
    """Fine-tune the top `depth` backbone layers, or re-load the saved result if the split is unchanged."""
    path = config.MODELS_DIR / f"cnn_ft{depth}.keras"
    if not force:
        m = load_cnn(path, data)
        if m is not None:
            _compile(m, 1e-5)
            return m, None, "loaded from cache"
    m = build_cnn(**head_model_cfg, head=head)
    h = finetune(m, data, depth=depth, epochs=epochs)
    save_cnn(m, path, data, depth=depth, tag=tag, val_acc=max(h.history["val_accuracy"]))
    return m, h, "trained"


# ----------------------------------------------------------------------------- evaluation helpers
def predict_probs(model, X, batch_size=64):
    return model.predict(X, batch_size=batch_size, verbose=0)


def evaluate(model, X, y, title, plot=True):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    prob = predict_probs(model, X); pred = prob.argmax(1)
    print(f"{title}: accuracy = {accuracy_score(y, pred):.4f} | macro-F1 = {f1_score(y, pred, average='macro'):.4f}")
    labels = list(range(NUM_CLASSES))
    print(classification_report(y, pred, labels=labels, target_names=CATEGORIES, digits=3, zero_division=0))
    cm = confusion_matrix(y, pred, labels=labels)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ConfusionMatrixDisplay(cm, display_labels=CATEGORIES).plot(ax=ax[0], xticks_rotation=60, colorbar=False, cmap="Blues")
        ax[0].set_title(f"{title} - counts")
        ConfusionMatrixDisplay(cm / np.maximum(cm.sum(1, keepdims=True), 1), display_labels=CATEGORIES).plot(
            ax=ax[1], xticks_rotation=60, colorbar=False, cmap="Blues", values_format=".2f")
        ax[1].set_title(f"{title} - row-normalised (recall)")
        plt.tight_layout(); plt.show()
    return pred, prob

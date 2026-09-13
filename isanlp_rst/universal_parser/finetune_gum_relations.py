"""Fine-tune the released UniRST relation head on native GUM labels."""

import json
import pickle
import random
from pathlib import Path

import fire
import numpy as np
import torch

from isanlp_rst.universal_parser.data_manager import DataManager
from isanlp_rst.universal_parser.predictor import PredictorUniRST
from isanlp_rst.universal_parser.relation_finetuning import (
    freeze_except_relation_head,
    replace_with_gum_fine_head,
)
from isanlp_rst.universal_parser.src.parser.data import RelationTableGUMFine
from isanlp_rst.universal_parser.src.parser.training_manager import TrainingManager


RELEASED_MODEL = 'tchewik/isanlp_rst_v3'
RELEASED_REVISION = '9407970f1d9d2435b5f875a0cd14293a16646304'
GUM_INVENTORY = 'eng.erst.gum'


def _load_data_manager(path, data_root):
    path = Path(path)
    requested_root = Path(data_root).expanduser().resolve()
    if path.exists():
        with path.open('rb') as stream:
            manager = pickle.load(stream)
        # Pickles include paths from the machine that built the cache. Always
        # rebase them to this invocation's data root instead of silently using
        # stale relative or absolute locations.
        manager.data_root = requested_root
        manager.input_path = requested_root / 'gum_rs3'
        manager.output_path = requested_root / 'gum_fine_prepared'
        if not manager.output_path.is_dir() or not any(manager.output_path.glob('*.pkl')):
            raise FileNotFoundError(
                f'Cached manager {path} was rebased to {manager.output_path}, but no prepared '
                'GUM documents were found there. Rebuild the cache for this --data-root.'
            )
    else:
        manager = DataManager('GUM', relation_granularity='fine', data_root=requested_root)
        manager.from_rs3()
        path.parent.mkdir(parents=True, exist_ok=True)
        manager.save(path)
    if getattr(manager, 'relation_granularity', None) != 'fine':
        raise ValueError(f'{path} is not a fine-grained GUM data manager')
    return manager


def _class_weights(data, classes, device):
    counts = np.bincount(
        [label for document in data.relation_label for label in document],
        minlength=classes,
    )
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f'Training split has no examples for relation classes {missing}')
    weights = 1.0 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(
    data_manager_path='data/data_manager_gum_fine.pickle',
    data_root='data',
    save_dir='saves',
    run_name='gum_v11_1_fine_relations',
    model_name=RELEASED_MODEL,
    revision=RELEASED_REVISION,
    cuda_device=-1,
    batch_size=1,
    eval_size=1,
    epochs=20,
    lr=1e-4,
    weight_decay=1e-2,
    patience=4,
    grad_norm=1.0,
    grad_clipping_value=10.0,
    seed=42,
    use_amp=False,
):
    """Train only the 50-way GUM relation classifier and test once at the end."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_device >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError('cuda_device was requested, but CUDA is unavailable')
        torch.cuda.manual_seed_all(seed)

    manager = _load_data_manager(data_manager_path, data_root)
    train_data, dev_data, test_data = manager.get_data(lang='en')

    predictor = PredictorUniRST(
        hf_model_name=model_name,
        hf_model_version=revision,
        relinventory=GUM_INVENTORY,
        cuda_device=cuda_device,
    )
    replace_with_gum_fine_head(predictor.model)
    freeze_except_relation_head(predictor.model)
    # Source labels already index RelationTableGUMFine; do not remap them into
    # the released union vocabulary. Tokenization retains GUM dataset index 1.
    predictor.label_maps = None

    train_data = predictor.tokenize(train_data)
    dev_data = predictor.tokenize(dev_data)
    test_data = predictor.tokenize(test_data)
    predictor.model.label_weights = [
        _class_weights(train_data, len(RelationTableGUMFine), predictor._cuda_device)
    ]

    provenance = {
        'task': 'gum_fine_relation_head',
        'base_model': model_name,
        'base_revision': revision,
        'gum_version': '11.1.0',
        'relation_inventory': list(RelationTableGUMFine),
        'selection_metric': 'gs_val_f1_rel',
        'test_policy': 'evaluated once after validation-selected training',
        'seed': seed,
        'artifact_format': 1,
    }
    trainer = TrainingManager(
        predictor.model, [train_data], [dev_data], [test_data],
        batch_size=batch_size, eval_size=eval_size, epochs=epochs,
        lr=lr, transformer_lr_multiplier=0, lr_decay_epoch=1000, lr_decay=1.0,
        weight_decay=weight_decay, grad_norm=grad_norm,
        grad_clipping_value=grad_clipping_value, patience=patience,
        use_micro_f1=True, use_dwa_loss=False, dwa_bs=max(batch_size, 1),
        save_dir=save_dir, use_amp=use_amp, run_name=run_name, config=provenance,
        loss_mode='relation', selection_metric='gs_val_f1_rel',
        evaluate_test_each_epoch=False, checkpoint_scope='relation_head',
    )
    run_dir = trainer.save_dir
    (run_dir / 'relation_table_eng.erst.gum.txt').write_text(
        '\n'.join(RelationTableGUMFine) + '\n', encoding='utf8')

    best_metrics = trainer.train()
    head_path = run_dir / 'best_relation_head.pt'
    predictor.model.label_classifier.load_state_dict(
        torch.load(head_path, map_location=predictor._cuda_device, weights_only=True)
    )
    test_metrics = trainer.evaluate_test()
    with (run_dir / 'test_metrics.json').open('w', encoding='utf8') as stream:
        json.dump(test_metrics, stream, indent=2, sort_keys=True, default=float)
    return {'run_dir': str(run_dir), 'best_validation': best_metrics, 'test': test_metrics}


if __name__ == '__main__':
    fire.Fire(train)

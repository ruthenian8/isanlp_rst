"""Utilities for adapting a released UniRST relation head."""

from typing import Dict, Iterable

import torch

from isanlp_rst.universal_parser.src.parser.data import RelationTableGUMFine
from isanlp_rst.universal_parser.src.parser.modules import DefaultLabelClassifier


def gum_fine_to_released_coarse(label: str) -> str:
    """Map a GUM fine relation/nuclearity class to its released coarse row."""

    relation, separator, nuclearity = label.rpartition('_')
    if separator == '' or nuclearity.upper() not in {'NN', 'NS', 'SN'}:
        raise ValueError(f'Invalid GUM relation label: {label!r}')

    coarse_relation = relation.lower()
    if coarse_relation != 'same-unit':
        coarse_relation = coarse_relation.split('-', 1)[0]
    if coarse_relation == 'contingency':
        coarse_relation = 'condition'
    elif coarse_relation == 'topic' and nuclearity.upper() == 'NS':
        # GUM V11.1 contains two topic-solutionhood_NS decisions, while the
        # released coarse inventory only exposes topic_SN. The released data
        # loader routed this exceptional NS form to condition_NS.
        coarse_relation = 'condition'
    return f'{coarse_relation}_{nuclearity.lower()}'


def _copy_output_row(source, target, source_index: int, target_index: int) -> None:
    target.weight_left.weight[target_index].copy_(
        source.weight_left.weight[source_index]
    )
    target.weight_right.weight[target_index].copy_(
        source.weight_right.weight[source_index]
    )
    target.weight_bilateral.weight[target_index].copy_(
        source.weight_bilateral.weight[source_index]
    )
    if source.weight_bilateral.bias is not None:
        target.weight_bilateral.bias[target_index].copy_(
            source.weight_bilateral.bias[source_index]
        )


def replace_with_gum_fine_head(model, fine_labels: Iterable[str] = RelationTableGUMFine):
    """Replace a released masked-union head with a GUM fine-grained head.

    The shared left/right projections are retained. Each fine output row is
    initialized from its corresponding coarse GUM row, so adaptation starts
    from the released classifier rather than a random relation head.
    """

    if not hasattr(model, 'label_classifier'):
        raise ValueError('Expected a released masked-union model with label_classifier')

    source = model.label_classifier
    fine_labels = list(fine_labels)
    source_vocab: Dict[str, int] = {
        label.lower(): index for index, label in enumerate(model.relation_vocab)
    }
    missing = sorted({
        gum_fine_to_released_coarse(label)
        for label in fine_labels
        if gum_fine_to_released_coarse(label) not in source_vocab
    })
    if missing:
        raise ValueError(f'Released checkpoint is missing coarse GUM rows: {missing}')

    device = source.labelspace_left.weight.device
    target = DefaultLabelClassifier(
        input_size=source.input_size,
        hidden_size=source.hidden_size,
        classes_number=len(fine_labels),
        bias=source.weight_bilateral.bias is not None,
        dropout=source.dropout.p,
        cuda_device=device,
    )

    with torch.no_grad():
        target.labelspace_left.weight.copy_(source.labelspace_left.weight)
        target.labelspace_right.weight.copy_(source.labelspace_right.weight)
        for target_index, label in enumerate(fine_labels):
            source_index = source_vocab[gum_fine_to_released_coarse(label)]
            _copy_output_row(source, target, source_index, target_index)

    model.label_classifier = target
    model.relation_vocab = [label.lower() for label in fine_labels]
    model.relation_tables = [fine_labels]
    model.classes_numbers = [len(fine_labels)]
    model.dataset_masks = [torch.ones(len(fine_labels), dtype=torch.bool, device=device)]
    # Retain the released dataset indices so GUM continues to select segmenter
    # 1, while routing every dataset index to this single relation classifier.
    model.dataset2classifier = [0 for _ in model.segmenters]
    return target


def freeze_except_relation_head(model) -> None:
    """Freeze UniRST and leave only its relation classifier trainable."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.label_classifier.parameters():
        parameter.requires_grad = True


def set_relation_finetuning_mode(model) -> None:
    """Disable frozen-model dropout while retaining classifier dropout."""

    model.eval()
    model.label_classifier.train()


def load_gum_fine_head(model, checkpoint, map_location='cpu') -> None:
    """Adapt a released model and load a saved GUM fine-head checkpoint."""

    replace_with_gum_fine_head(model)
    state_dict = torch.load(checkpoint, map_location=map_location, weights_only=True)
    model.label_classifier.load_state_dict(state_dict)

"""Utilities for adapting a released UniRST relation head."""

from typing import Dict, Iterable, List

import torch

from isanlp_rst.universal_parser.src.parser.data import (
    RelationTableGUMFine,
    RelationTableRSTDTFine,
)
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


_RSTDT_FINE_TO_COARSE = {
    **{name: 'attribution' for name in ('attribution',)},
    **{name: 'background' for name in ('background', 'circumstance')},
    **{name: 'cause' for name in ('cause', 'cause-result', 'result', 'consequence')},
    **{name: 'comparison' for name in ('comparison', 'preference', 'analogy', 'proportion')},
    **{name: 'condition' for name in ('condition', 'hypothetical', 'contingency', 'otherwise')},
    **{name: 'contrast' for name in ('contrast', 'concession', 'antithesis')},
    **{name: 'enablement' for name in ('purpose', 'enablement')},
    **{name: 'evaluation' for name in ('evaluation', 'interpretation', 'conclusion', 'comment')},
    **{name: 'explanation' for name in ('evidence', 'explanation-argumentative', 'reason')},
    **{name: 'joint' for name in ('list', 'disjunction')},
    **{name: 'manner-means' for name in ('manner', 'means')},
    **{name: 'topic-comment' for name in (
        'problem-solution', 'question-answer', 'statement-response',
        'topic-comment', 'comment-topic', 'rhetorical-question')},
    **{name: 'summary' for name in ('summary', 'restatement')},
    **{name: 'temporal' for name in (
        'temporal-before', 'temporal-after', 'temporal-same-time',
        'sequence', 'inverted-sequence')},
    **{name: 'topic-change' for name in ('topic-shift', 'topic-drift')},
    'textual-organization': 'textual-organization',
    'same-unit': 'same-unit',
}


def rstdt_fine_to_released_coarse(label: str) -> str:
    """Map a suffix-merged native RST-DT class to the released coarse row."""

    relation, separator, nuclearity = label.rpartition('_')
    if separator == '' or nuclearity.upper() not in {'NN', 'NS', 'SN'}:
        raise ValueError(f'Invalid RST-DT relation label: {label!r}')
    base_relation = relation.lower()
    if base_relation.endswith('-mn'):
        base_relation = base_relation[:-3]
    if base_relation.startswith('elaboration-') or base_relation in {'example', 'definition'}:
        coarse_relation = 'elaboration'
    else:
        try:
            coarse_relation = _RSTDT_FINE_TO_COARSE[base_relation]
        except KeyError as error:
            raise ValueError(f'Unknown native RST-DT relation: {relation!r}') from error
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


def _copy_mean_output_row(
    source, target, source_indices: List[int], target_index: int
) -> None:
    """Initialize a missing nuclearity row from the same existing relation."""

    for attribute in ('weight_left', 'weight_right', 'weight_bilateral'):
        source_layer = getattr(source, attribute)
        target_layer = getattr(target, attribute)
        target_layer.weight[target_index].copy_(
            source_layer.weight[source_indices].mean(dim=0)
        )
    if source.weight_bilateral.bias is not None:
        target.weight_bilateral.bias[target_index].copy_(
            source.weight_bilateral.bias[source_indices].mean(dim=0)
        )


def _source_rows_for_label(label: str, source_vocab: Dict[str, int]) -> List[int]:
    """Resolve an exact row, or rows for the same relation across nuclearities."""

    normalized = label.lower()
    if normalized in source_vocab:
        return [source_vocab[normalized]]

    relation, separator, nuclearity = normalized.rpartition('_')
    if separator == '' or nuclearity not in {'nn', 'ns', 'sn'}:
        return []
    return [
        source_vocab[candidate]
        for candidate in (f'{relation}_nn', f'{relation}_ns', f'{relation}_sn')
        if candidate in source_vocab
    ]


def _replace_with_fine_head(model, fine_labels, coarse_label):
    """Replace a released masked-union head and initialize rows from coarse labels.

    The shared left/right projections are retained. Each fine output row is
    initialized from its corresponding released coarse row, so adaptation starts
    from the released classifier rather than a random relation head.
    """

    if not hasattr(model, 'label_classifier'):
        raise ValueError('Expected a released masked-union model with label_classifier')

    source = model.label_classifier
    fine_labels = list(fine_labels)
    source_vocab: Dict[str, int] = {
        label.lower(): index for index, label in enumerate(model.relation_vocab)
    }
    source_rows = {
        label: _source_rows_for_label(coarse_label(label), source_vocab)
        for label in fine_labels
    }
    missing = sorted({coarse_label(label) for label in fine_labels if not source_rows[label]})
    if missing:
        raise ValueError(
            'Released checkpoint has no rows for these coarse relations: '
            f'{missing}'
        )

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
            indices = source_rows[label]
            if len(indices) == 1:
                _copy_output_row(source, target, indices[0], target_index)
            else:
                _copy_mean_output_row(source, target, indices, target_index)

    model.label_classifier = target
    model.relation_vocab = [label.lower() for label in fine_labels]
    model.relation_tables = [fine_labels]
    model.classes_numbers = [len(fine_labels)]
    model.dataset_masks = [torch.ones(len(fine_labels), dtype=torch.bool, device=device)]
    # Retain released dataset indices so inference continues to select the
    # inventory-specific segmenter while every route uses this classifier.
    model.dataset2classifier = [0 for _ in model.segmenters]
    # Parsing losses index these weights by dataset index. Older masked-union
    # models incorrectly initialized this list from the number of classifiers
    # (one) rather than the number of dataset routes.
    model.corpora_weights = [1.0 for _ in model.dataset2classifier]
    if hasattr(model, 'encoder'):
        model.encoder.corpora_weights = model.corpora_weights
    return target


def replace_with_gum_fine_head(model, fine_labels: Iterable[str] = RelationTableGUMFine):
    """Install a native GUM head initialized from released coarse rows."""

    return _replace_with_fine_head(model, fine_labels, gum_fine_to_released_coarse)


def replace_with_rstdt_fine_head(
    model, fine_labels: Iterable[str] = RelationTableRSTDTFine
):
    """Install a native suffix-merged RST-DT head from released coarse rows."""

    return _replace_with_fine_head(model, fine_labels, rstdt_fine_to_released_coarse)


def freeze_except_relation_head(model) -> None:
    """Freeze UniRST and leave only its relation classifier trainable."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.label_classifier.parameters():
        parameter.requires_grad = True


def configure_dropout(model, classifier=None, transformer=None, segmenter=None):
    """Apply optional targeted dropout overrides and report effective values."""

    for name, probability in (
        ('classifier_dropout', classifier),
        ('transformer_dropout', transformer),
        ('segmenter_dropout', segmenter),
    ):
        if probability is not None and not 0 <= probability < 1:
            raise ValueError(f'{name} must be in [0, 1), got {probability}')

    if classifier is not None:
        model.label_classifier.dropout.p = classifier
    transformer_dropouts = []
    transformer_module = getattr(getattr(model, 'encoder', None), 'transformer', None)
    if transformer_module is not None:
        transformer_dropouts = [
            module for module in transformer_module.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        if transformer is not None:
            for module in transformer_dropouts:
                module.p = transformer

    segmenter_dropouts = [
        item.dropout for item in getattr(model, 'segmenters', [])
        if isinstance(getattr(item, 'dropout', None), torch.nn.Dropout)
    ]
    if segmenter is not None:
        for module in segmenter_dropouts:
            module.p = segmenter

    encoder = getattr(model, 'encoder', None)
    return {
        'classifier': model.label_classifier.dropout.p,
        'encoder': getattr(getattr(encoder, 'dropout', None), 'p', None),
        'edu': getattr(getattr(encoder, 'edu_dropout', None), 'p', None),
        'segmenter': sorted({module.p for module in segmenter_dropouts}),
        'transformer': sorted({module.p for module in transformer_dropouts}),
    }


def set_relation_finetuning_mode(model) -> None:
    """Disable frozen-model dropout while retaining classifier dropout."""

    model.eval()
    model.label_classifier.train()


def load_gum_fine_head(model, checkpoint, map_location='cpu') -> None:
    """Adapt a released model and load a saved GUM fine-head checkpoint."""

    replace_with_gum_fine_head(model)
    state_dict = torch.load(checkpoint, map_location=map_location, weights_only=True)
    model.label_classifier.load_state_dict(state_dict)


def load_rstdt_fine_head(model, checkpoint, map_location='cpu') -> None:
    """Adapt a released model and load a saved RST-DT fine-head checkpoint."""

    replace_with_rstdt_fine_head(model)
    state_dict = torch.load(checkpoint, map_location=map_location, weights_only=True)
    model.label_classifier.load_state_dict(state_dict)

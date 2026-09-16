"""Prepare converted RST-DT RS3 trees for UniRST without collapsing labels."""

from pathlib import Path


def _render_relation(node, edu_indices):
    relation = node.label()
    roles = [child.label() for child in node]

    def render_argument(wrapper):
        child = wrapper[0]
        if isinstance(child, str):
            return f"(EDU {edu_indices[str(wrapper.root_id)]})"
        return _render_relation(child, edu_indices)

    arguments = [render_argument(child) for child in node]
    if len(arguments) == 2:
        nuclearity = "".join(roles)
        if nuclearity not in {"NN", "NS", "SN"}:
            raise ValueError(
                f"Unsupported nuclearity {nuclearity!r} for relation {relation!r}"
            )
        return f"({nuclearity}-{relation} {arguments[0]} {arguments[1]})"

    if len(arguments) > 2 and set(roles) == {"N"}:
        # Match the parser's binary representation with deterministic right
        # branching for multinuclear schemas.
        result = arguments[-1]
        for argument in reversed(arguments[:-1]):
            result = f"(NN-{relation} {argument} {result})"
        return result

    raise ValueError(
        f"Unsupported {len(arguments)}-argument schema for relation {relation!r}: {roles}"
    )


def convert_rs3_document(source, output_dir):
    """Write one ``.lisp``/``.edus`` pair using ``rstconverter``'s RS3 tree."""

    try:
        from rstconverter import read_rs3tree
    except ImportError as error:
        raise ImportError(
            "Fine-grained RST-DT preparation requires the rstconverter package"
        ) from error

    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document = read_rs3tree(str(source))
    edu_indices = {
        str(source_id): index
        for index, source_id in enumerate(document.edus, start=1)
    }
    basename = source.name[:-4] if source.name.endswith(".rs3") else source.name
    if basename.endswith(".out"):
        basename = basename[:-4]
    (output_dir / f"{basename}.lisp").write_text(
        _render_relation(document.tree, edu_indices), encoding="utf8"
    )
    edu_lines = [
        " ".join(document.elem_dict[str(source_id)]["text"].split())
        for source_id in document.edus
    ]
    (output_dir / f"{basename}.edus").write_text(
        "\n".join(edu_lines) + "\n", encoding="utf8"
    )

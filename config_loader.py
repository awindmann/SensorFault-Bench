# config_loader.py
"""Single module to load defaults from YAML + CLI overrides."""

import argparse
from pathlib import Path

import yaml


def _is_scope_list_key(key) -> bool:
    return str(key).upper() in {"MODEL", "METHOD"}


def load_defaults(path: Path = Path("configs/defaults.yaml")) -> dict:
    """Load defaults.yaml. Raise FileNotFoundError if missing."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_benchmark_scope(path: Path = Path("configs/benchmark_scope.yaml")) -> dict:
    """Load the benchmark-scope manifest. Raise FileNotFoundError if missing."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_core_figure_registry(
    path: Path = Path("configs/reporting/core_figures.yaml"),
) -> dict:
    """Load the core-figure registry YAML. Raise FileNotFoundError if missing."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_parsed_core_figure_registry(
    path: Path = Path("configs/reporting/core_figures.yaml"),
):
    """Load and parse the dedicated core-figure registry config."""
    from utils.parsing import parse_core_figure_registry_config

    return parse_core_figure_registry_config(
        load_core_figure_registry(path),
        context=str(path),
    )


def load_hparams(path: Path = Path("configs/baseline_hparams.yaml")) -> dict:
    """Load baseline_hparams.yaml. Raise FileNotFoundError if missing."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset_windows(
    path: Path = Path("configs/dataset_windows.yaml"),
    *,
    defaults: dict,
) -> dict[str, dict[str, int]]:
    """Load dataset-specific window and batch-size defaults from YAML.

    ``required_datasets`` is derived from ``defaults['DATA_FILES']``,
    resolved to canonical dataset spec keys.
    """
    from data.datasets import resolve_with_defaults
    from utils.parsing import parse_dataset_window_defaults

    if "DATA_FILES" not in defaults:
        raise ValueError("defaults is missing required key 'DATA_FILES'.")
    if "DATA_ROOT" not in defaults:
        raise ValueError("defaults is missing required key 'DATA_ROOT'.")
    data_files = defaults["DATA_FILES"]
    if not isinstance(data_files, list) or len(data_files) == 0:
        raise ValueError("defaults['DATA_FILES'] must be a non-empty list.")

    specs = resolve_with_defaults(
        data_files,
        [],
        data_root=defaults["DATA_ROOT"],
    )
    canonical_keys = tuple(s.key for s in specs)

    with open(path) as f:
        raw = yaml.safe_load(f)
    return parse_dataset_window_defaults(
        raw,
        required_datasets=canonical_keys,
    )


def load_cli_help(path: Path = Path("configs/cli_help.yaml")):
    """Load the grouped help text of the command-line options."""
    from utils.parsing import parse_cli_help_config

    with open(path) as f:
        return parse_cli_help_config(yaml.safe_load(f), key=str(path))


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    """Build argparse with defaults from YAML and help text from configs/cli_help.yaml.

    Options appear in the groups of configs/cli_help.yaml. An option without a
    help entry raises, so every option documents itself in ``--help``.
    """
    help_groups = load_cli_help()
    documented = {key for group in help_groups for key in group.options}
    undocumented = [key for key in defaults if key not in documented]
    if undocumented:
        raise ValueError(
            "configs/cli_help.yaml has no help entry for option(s): "
            f"{', '.join(undocumented)}. Add one entry per option."
        )
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for group in help_groups:
        keys = [key for key in group.options if key in defaults]
        if not keys:
            continue
        argument_group = parser.add_argument_group(group.title, group.description)
        for key in keys:
            _add_default_argument(argument_group, key, defaults[key], help_text=group.options[key])
    return parser


def _add_default_argument(parser, key: str, value, *, help_text: str) -> None:
    arg_name = f"--{key.lower().replace('_', '-')}"
    if _is_scope_list_key(key):
        parser.add_argument(arg_name, nargs="+", default=value, help=help_text)
    elif isinstance(value, bool):
        parser.add_argument(
            arg_name, action=argparse.BooleanOptionalAction, default=value, help=help_text,
        )
    elif isinstance(value, list):
        parser.add_argument(arg_name, nargs="+", default=value, help=help_text)
    elif value is None:
        parser.add_argument(arg_name, default=None, help=help_text)
    else:
        parser.add_argument(arg_name, type=type(value), default=value, help=help_text)


def parse_explicit_cli_overrides(param_defaults: dict, extra_args=None) -> dict:
    """Parse CLI args and return only values explicitly provided by the user.

    Uses `parse_known_args` so unrelated flags are ignored.
    """
    parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    for key, value in param_defaults.items():
        arg_name = f"--{str(key).lower().replace('_', '-')}"
        if _is_scope_list_key(key):
            parser.add_argument(arg_name, nargs="+")
        elif isinstance(value, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction)
        elif isinstance(value, list):
            parser.add_argument(arg_name, nargs="*")
        elif value is None:
            parser.add_argument(arg_name)
        else:
            parser.add_argument(arg_name, type=type(value))
    namespace, _ = parser.parse_known_args(extra_args)
    return vars(namespace)


def load_config(extra_args=None) -> argparse.Namespace:
    """Load defaults, parse CLI, return merged config.

    Args:
        extra_args: Optional list of args to parse instead of sys.argv.
                   Useful for testing.
    """
    defaults = load_defaults()
    parser = build_parser(defaults)
    if extra_args is not None:
        return parser.parse_args(extra_args)
    return parser.parse_args()

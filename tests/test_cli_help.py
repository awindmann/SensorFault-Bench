"""Checks that every command-line option of the entry points carries help text."""

from __future__ import annotations

import pytest
import yaml

from config_loader import build_parser, load_cli_help, load_defaults
from pipelines.selection import (
    extract_recipe_defaults_for_scope,
    load_benchmark_recipe_specs_for_scope,
    merge_recipe_defaults_for_scope,
)
from utils.parsing import parse_cli_help_config


def _entrypoint_defaults(defaults: dict[str, object]) -> dict[str, object]:
    specs = load_benchmark_recipe_specs_for_scope(defaults)
    return merge_recipe_defaults_for_scope(defaults, extract_recipe_defaults_for_scope(specs))


def test_help_file_documents_exactly_the_entrypoint_options(registered_methods_defaults):
    # The help file documents the options of every registered method, whichever
    # of them BENCHMARK_METHODS selects.
    documented = [key for group in load_cli_help() for key in group.options]

    assert sorted(documented) == sorted(_entrypoint_defaults(registered_methods_defaults))


def test_help_output_shows_groups_help_text_and_defaults(registered_methods_defaults):
    defaults = _entrypoint_defaults(registered_methods_defaults)
    text = build_parser(defaults).format_help()

    for group in load_cli_help():
        assert f"{group.title}:" in text
    assert "Dataset keys to run." in text
    assert f"(default: {defaults['N_TEST_SAMPLES']})" in text


def test_build_parser_raises_for_an_option_without_help():
    with pytest.raises(ValueError, match="no help entry for option.*MY_NEW_OPTION"):
        build_parser({**load_defaults(), "MY_NEW_OPTION": 1})


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        ({"groups": []}, "non-empty list"),
        ({"groups": [{"title": "A", "options": {"SEED": "Seed."}, "extra": 1}]}, "unsupported key"),
        ({"groups": [{"title": "A", "options": {"SEED": " "}}]}, "SEED"),
        (
            {"groups": [{"title": "A", "options": {"SEED": "Seed."}}, {"title": "B", "options": {"SEED": "Seed."}}]},
            "more than once",
        ),
    ),
)
def test_help_config_rejects_malformed_entries(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_cli_help_config(yaml.safe_load(yaml.safe_dump(raw)))

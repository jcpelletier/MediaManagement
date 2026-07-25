import argparse
import pytest
import llm_deepseek
import Sort_TV


def test_llm_deepseek_default_model():
    """Verify that the DEFAULT_MODEL in llm_deepseek.py is set to deepseek-v4-flash."""
    assert llm_deepseek.DEFAULT_MODEL == "deepseek-v4-flash"


def test_sort_tv_model_defaults(monkeypatch):
    """Verify that Sort_TV.py's argument parser defaults to deepseek-v4-flash for both guided and blind models."""
    parsed_args = None
    original_parse_args = argparse.ArgumentParser.parse_args

    def mock_parse_args(self, args=None, namespace=None):
        nonlocal parsed_args
        parsed_args = original_parse_args(self, args, namespace)
        return parsed_args

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", mock_parse_args)
    # Mock DEEPSEEK_API_KEY so DeepSeekClient can be initialized without raising an error
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy_key_for_testing_defaults")

    # Run main with a non-existent directory (main will catch it and raise SystemExit,
    # but parse_args will have already run successfully).
    with pytest.raises(SystemExit):
        Sort_TV.main(["--root", "/nonexistent/directory/for/test/defaults"])

    assert parsed_args is not None
    assert parsed_args.model == "deepseek-v4-flash"
    assert parsed_args.blind_model == "deepseek-v4-flash"

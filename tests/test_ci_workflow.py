# Tests for the CI workflow (.github/workflows/test.yml) added for story #24.
# These exercise the workflow file itself (structured text checks, no extra
# dependencies) so a broken or mis-triggered workflow is caught in the suite
# instead of at merge time.

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "test.yml"


def test_workflow_exists():
    assert WORKFLOW.is_file(), f"missing workflow file: {WORKFLOW}"


@pytest.fixture(scope="module")
def workflow_text():
    return WORKFLOW.read_text(encoding="utf-8")


def _on_section(workflow_text):
    # Everything between the top-level 'on:' and the top-level 'jobs:'.
    m = re.search(r"\bon:\n(.*?)\njobs:", workflow_text, re.DOTALL)
    assert m, "workflow must declare an 'on' trigger block before 'jobs:'"
    return m.group(1)


def test_workflow_runs_on_prs_and_pushes_to_main(workflow_text):
    # The workflow must trigger on pull requests to main and pushes to main.
    section = _on_section(workflow_text)
    for trigger in ("pull_request", "push"):
        pattern = rf"{trigger}:\n(.*?branches:\s*\[([^\]]*)\])"
        m = re.search(pattern, section, re.DOTALL)
        assert m, f"expected trigger '{trigger}' under 'on' targeting a branch list"
        branches = [b.strip().strip("\"'") for b in m.group(2).split(",")]
        assert "main" in branches, f"trigger '{trigger}' must target the 'main' branch"


def test_workflow_uses_python_312_and_pytest(workflow_text):
    # The job must set up Python 3.12, install the test deps, and run pytest.
    assert re.search(r"python-version:\s*['\"]?3\.12", workflow_text), (
        "workflow must set up Python 3.12"
    )
    assert re.search(r"pip install\s+-r tests/requirements\.txt", workflow_text), (
        "workflow must install tests/requirements.txt"
    )
    assert re.search(r"run:\s*pytest", workflow_text), "workflow must run pytest"


def test_workflow_job_is_named_test(workflow_text):
    # The single job must be named 'test'.
    m = re.search(r"^jobs:\s*\n(\s+)(\w+):", workflow_text, re.MULTILINE)
    assert m, "workflow must declare at least one job"
    assert m.group(2) == "test", "the job must be named 'test'"


def test_suite_is_clean_with_no_network_or_api_keys():
    """Guard: no other test may require the live media library.

    If a test needs media, network, or API keys it must skip (with a reason)
    rather than fail, so a clean CI environment stays green.
    """
    tests_dir = Path(__file__).resolve().parent
    for path in sorted(tests_dir.glob("test_*.py")):
        # This file quotes the media root inside an assertion; skip it.
        if path.name == "test_ci_workflow.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "/mnt/media" not in text, (
            f"{path.name} references a live media path; such tests must be skipped in CI"
        )

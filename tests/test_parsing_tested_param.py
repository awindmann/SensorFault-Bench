import pytest

from utils.parsing import require_tested_param


def test_require_tested_param_accepts_true():
    assert require_tested_param({"tested": "true"}, run_id="run_1") is True


def test_require_tested_param_accepts_false():
    assert require_tested_param({"tested": "false"}, run_id="run_2") is False


def test_require_tested_param_raises_when_missing():
    with pytest.raises(ValueError, match="missing required 'tested' param"):
        require_tested_param({}, run_id="run_3")


def test_require_tested_param_raises_on_invalid_value():
    with pytest.raises(ValueError, match="invalid 'tested' param"):
        require_tested_param({"tested": "not-a-bool"}, run_id="run_4")

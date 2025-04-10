import pytest

# Tests for each unique threshold in reward_thresholds
def test_expected_result_one_env_3432807():
    """Test if the one environment performance meets the 3432807.680391572 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    assert one_env_performance >= 3432807.680391572

def test_expected_result_one_env_1e8():
    """Test if the one environment performance meets the 1e8 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    assert one_env_performance >= 1e8

def test_expected_result_one_env_1e9():
    """Test if the one environment performance meets the 1e9 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    assert one_env_performance >= 1e9

def test_expected_result_one_env_1e10():
    """Test if the one environment performance meets the 1e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    assert one_env_performance >= 1e10

def test_expected_result_one_env_2e10():
    """Test if the one environment performance meets the 2e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    assert one_env_performance >= 2e10

def test_expected_result_one_env_5e10():
    """Test if the one environment performance meets the 5e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    assert one_env_performance >= 5e10

# Tests for each unique threshold in reward_dr_thresholds
def test_expected_result_dr_env_1e10():
    """Test if the DR environment performance meets the 1e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    dr_env_performance = lines[1]
    assert dr_env_performance >= 1e10

def test_expected_result_dr_env_2e10():
    """Test if the DR environment performance meets the 2e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    dr_env_performance = lines[1]
    assert dr_env_performance >= 2e10

def test_expected_result_dr_env_5e10():
    """Test if the DR environment performance meets the 5e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    dr_env_performance = lines[1]
    assert dr_env_performance >= 5e10

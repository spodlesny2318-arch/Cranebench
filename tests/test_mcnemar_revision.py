import numpy as np
import pytest
from cranebench.stats import mcnemar

@pytest.mark.parametrize('a,b', [([0, 0], [0, 0]), ([1, 0], [1, 0]), ([], [])])
def test_no_discordance_has_unit_p(a, b):
    assert mcnemar(a, b)['p'] == 1
    assert mcnemar(a, b)['statistic'] == 0

def test_ten_one_way_discordances_exact_probability():
    result = mcnemar(np.ones(10), np.zeros(10))
    assert result == {'a_only': 10, 'b_only': 0, 'statistic': 0., 'p': 2 / 2**10}
    assert mcnemar(np.zeros(10), np.ones(10))['p'] == result['p']

@pytest.mark.parametrize('a,b', [([np.nan], [0]), ([2], [1]), ([0], [1, 0]), ([[0]], [[1]])])
def test_invalid_pairs_rejected(a, b):
    with pytest.raises(ValueError):
        mcnemar(a, b)

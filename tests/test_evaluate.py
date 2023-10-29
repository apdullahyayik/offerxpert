"""Tests for `evaluate` module."""
# pylint: disable=no-self-use,too-few-public-methods

import pytest

from offerexpert.evaluate import isin_expected_range


class TestEvaluate:
    """Test evaluate module"""

    @pytest.mark.parametrize(
        ("value", "mean", "standard_deviation", "expected"),
        [
            pytest.param(
                10,
                5,
                3,
                True,
            ),
            pytest.param(
                20,
                15,
                1,
                False,
            ),
            pytest.param(
                500,
                200,
                20,
                False,
            ),
            pytest.param(
                "",
                20,
                10,
                True,
            ),
            pytest.param(
                None,
                20,
                10,
                True,
            ),
            pytest.param(
                100,
                None,
                None,
                True,
            ),
        ],
    )
    def test_isin_expected_range(
        self,
        value: float | str | None,
        mean: float | str | None,
        standard_deviation: float | str | None,
        expected: bool,
    ):
        """Test `isin_expected_range` function."""
        output = isin_expected_range(value, mean, standard_deviation)
        assert output == expected

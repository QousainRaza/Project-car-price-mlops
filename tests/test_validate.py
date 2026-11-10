import pandas as pd
import pytest

try:
    from pandera.errors import SchemaErrors
except ImportError:  # pragma: no cover
    from pandera.pandas import SchemaErrors

from src.validate import validate

GOOD = dict(price=[2_500_000.0], year=[2018], mileage=[60000.0], brand=["Toyota"],
            car_model=["Toyota Corolla"], city=["Lahore"], engine_cc=[1300.0],
            fuel=["Petrol"], transmission=["Manual"])


def test_good_row_passes():
    validate(pd.DataFrame(GOOD))


def test_negative_price_fails():
    with pytest.raises(SchemaErrors):
        validate(pd.DataFrame({**GOOD, "price": [-5.0]}))


def test_year_1900_fails():
    with pytest.raises(SchemaErrors):
        validate(pd.DataFrame({**GOOD, "year": [1900]}))

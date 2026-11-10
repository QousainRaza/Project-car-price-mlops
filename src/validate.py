"""Stage 1: data checks. Bad data stops the pipeline before training."""
try:                                    # newer pandera
    import pandera.pandas as pa
    from pandera.pandas import Check, Column
except ImportError:                     # older pandera
    import pandera as pa
    from pandera import Check, Column

from src.config import ENGINE_MAX, ENGINE_MIN, YEAR_MAX, YEAR_MIN

schema = pa.DataFrameSchema(
    {
        "price": Column(float, Check.in_range(50_000, 1_000_000_000)),
        "year": Column(int, Check.in_range(YEAR_MIN, YEAR_MAX)),
        "mileage": Column(float, Check.in_range(0, 1_000_000)),
        "brand": Column(str, nullable=False),
        "car_model": Column(str, nullable=False),
        "city": Column(str, nullable=False),
        "engine_cc": Column(float, Check.in_range(ENGINE_MIN, ENGINE_MAX)),
        "fuel": Column(str, nullable=False),
        "transmission": Column(str, nullable=False),
    },
    coerce=True,
)


def validate(df):
    """Return the validated frame, or raise SchemaErrors listing ALL problems."""
    return schema.validate(df, lazy=True)

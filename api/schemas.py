from pydantic import BaseModel, Field


class CarIn(BaseModel):
    year: int = Field(ge=1970, le=2026)
    mileage: float = Field(ge=0, le=1_000_000)
    brand: str = Field(min_length=1)          # e.g. Toyota
    car_model: str = Field(min_length=1)      # e.g. Toyota Corolla
    city: str = Field(min_length=1)
    engine_cc: float = Field(ge=300, le=8000)
    fuel: str = Field(min_length=1)
    transmission: str = Field(min_length=1)


class PriceOut(BaseModel):
    predicted_price: float
    model_version: str

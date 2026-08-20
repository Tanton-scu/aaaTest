from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.-]+$")
    generations: int = Field(default=4, ge=1, le=20)
    population_size: int = Field(default=10, ge=2, le=40)
    candidate_budget: int = Field(default=20, ge=1, le=200)
    random_seed: int = Field(default=2024, ge=0, le=2147483647)

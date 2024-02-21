import pydantic as _pydantic
from typing import Optional
from fastapi import UploadFile


class _PromptBase(_pydantic.BaseModel):
    seed: Optional[float] = -1
    num_inference_steps: int = 50
    guidance_scale: float = 3
    strength: float = 0.55


class ImageCreate(_PromptBase):
    current_gender: str = 'Undefined'
    encoded_base_img: UploadFile
    img_width: int = 512

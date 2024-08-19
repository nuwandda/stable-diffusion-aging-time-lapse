import schemas as _schemas

import torch 
from diffusers import StableDiffusionImg2ImgPipeline
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from utils import set_seed, gender
import numpy as np
from pkg_resources import parse_version


# Get the token from HuggingFace 
"""
Note: make sure .env exist and contains your token
"""
MODEL_PATH = 'SG161222/Realistic_Vision_V6.0_B1_noVAE'
 

def create_pipeline(model_path):
    # Create the pipe 
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path, 
        revision="fp16", 
        torch_dtype=torch.float16
        )
    
    # pipe.load_lora_weights(pretrained_model_name_or_path_or_dict="weights/lora_disney.safetensors", adapter_name="disney")

    if torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe.to(device)
    
    return pipe


pipe = create_pipeline(MODEL_PATH)


async def generate_image(imgPrompt: _schemas.ImageCreate) -> Image:
    generator = torch.Generator().manual_seed(set_seed()) if float(imgPrompt.seed) == -1 else torch.Generator().manual_seed(int(imgPrompt.seed))
    request_object_content = await imgPrompt.encoded_base_img.read()
    init_image = Image.open(BytesIO(request_object_content))
    aspect_ratio = init_image.width / init_image.height
    target_height = round(imgPrompt.img_width / aspect_ratio)
    
    # Resize the image
    if parse_version(Image.__version__) >= parse_version('9.5.0'):
        resized_image = init_image.resize((imgPrompt.img_width, target_height), Image.LANCZOS)
    else:
        resized_image = init_image.resize((imgPrompt.img_width, target_height), Image.ANTIALIAS)
    
    # Predict gender if necessary, then add it to the prompt
    if imgPrompt.current_gender == 'Undefined':
        gender_result, _ = gender(np.array(resized_image))
    else:
        gender_result = imgPrompt.current_gender
    
    print(gender_result.lower())
    if gender_result == 'Multiple faces':
        return 'There should be single face in the image.'
    
    child_gender = 'boy' if gender_result.lower() == 'male' else 'girl'
    
    final_prompt_70 = """photo of a 70 years old {}, detailed (wrinkles, blemishes, folds, moles, veins, pores, skin imperfections:1.1),
    highly detailed glossy eyes, specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, 
    film grain, centered, Fujifilm XT3, crystal clear""".format(gender_result.lower())
    final_prompt_50 = """photo of a 50 years old {}, detailed (wrinkles, blemishes, folds, moles, veins, pores, skin imperfections:1.1),
    highly detailed glossy eyes, specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, 
    film grain, centered, Fujifilm XT3, crystal clear""".format(gender_result.lower())
    final_prompt_30 = """photo of a 30 years old {}, detailed (wrinkles, blemishes, folds, moles, veins, pores, skin imperfections:1.1),
    highly detailed glossy eyes, specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, 
    film grain, centered, Fujifilm XT3, crystal clear""".format(gender_result.lower())
    final_prompt_10 = """photo of a 10 years old {}, detailed (wrinkles, blemishes, folds, moles, veins, pores, skin imperfections:1.1),
    highly detailed glossy eyes, specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, 
    film grain, centered, Fujifilm XT3, crystal clear""".format(child_gender)
    negative_prompt_male = """naked, nude, out of frame, tattoo, b&w, sepia,
    (blurry un-sharp fuzzy un-detailed skin:1.4), (twins:1.4), (geminis:1.4), (wrong eyeballs:1.1),
    (cloned face:1.1), (perfect skin:1.2), (mutated hands and fingers:1.3), disconnected hands,
    disconnected limbs, amputation, (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing,
    anime, doll, overexposed, photoshop, oversaturated:1.4)"""
    negative_prompt_female = """beard, moustache, naked, nude, out of frame, tattoo, b&w, sepia,
    (blurry un-sharp fuzzy un-detailed skin:1.4), (twins:1.4), (geminis:1.4), (wrong eyeballs:1.1),
    (cloned face:1.1), (perfect skin:1.2), (mutated hands and fingers:1.3), disconnected hands,
    disconnected limbs, amputation, (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing,
    anime, doll, overexposed, photoshop, oversaturated:1.4)"""
    negative_prompt_child = """beard, moustache, naked, nude, out of frame, tattoo, b&w, sepia,
    (blurry un-sharp fuzzy un-detailed skin:1.4), (twins:1.4), (geminis:1.4), (wrong eyeballs:1.1),
    (cloned face:1.1), (perfect skin:1.2), (mutated hands and fingers:1.3), disconnected hands,
    disconnected limbs, amputation, (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing,
    anime, doll, overexposed, photoshop, oversaturated:1.4)"""

    # 70 years old generation
    print('Generating 70 years old...')
    image_70: Image = pipe(final_prompt_70,
                                image=resized_image, strength=imgPrompt.strength,
                                negative_prompt=negative_prompt_male if gender_result.lower() == 'male' else negative_prompt_female, 
                                guidance_scale=imgPrompt.guidance_scale, 
                                num_inference_steps=imgPrompt.num_inference_steps, 
                                generator = generator,
                                cross_attention_kwargs={"scale": imgPrompt.strength}
                                ).images[0]

    if not image_70.getbbox():
        image_70: Image = pipe(final_prompt_70,
                                    image=resized_image, strength=imgPrompt.strength + 0.1,
                                    negative_prompt=negative_prompt_male if gender_result.lower() == 'male' else negative_prompt_female,
                                    guidance_scale=imgPrompt.guidance_scale, 
                                    num_inference_steps=imgPrompt.num_inference_steps, 
                                    generator = generator,
                                    cross_attention_kwargs={"scale": imgPrompt.strength}
                                    ).images[0]
        
    # 50 years old generation
    print('Generating 50 years old...')
    image_50: Image = pipe(final_prompt_50,
                                image=resized_image, strength=imgPrompt.strength,
                                negative_prompt=negative_prompt_male if gender_result.lower() == 'male' else negative_prompt_female, 
                                guidance_scale=imgPrompt.guidance_scale, 
                                num_inference_steps=imgPrompt.num_inference_steps, 
                                generator = generator,
                                cross_attention_kwargs={"scale": imgPrompt.strength}
                                ).images[0]

    if not image_50.getbbox():
        image_50: Image = pipe(final_prompt_50,
                                    image=resized_image, strength=imgPrompt.strength + 0.1,
                                    negative_prompt=negative_prompt_male if gender_result.lower() == 'male' else negative_prompt_female,
                                    guidance_scale=imgPrompt.guidance_scale, 
                                    num_inference_steps=imgPrompt.num_inference_steps, 
                                    generator = generator,
                                    cross_attention_kwargs={"scale": imgPrompt.strength}
                                    ).images[0]
        
    # 30 years old generation
    print('Generating 30 years old...')
    image_30: Image = pipe(final_prompt_30,
                                image=resized_image, strength=imgPrompt.strength,
                                negative_prompt=negative_prompt_male if gender_result.lower() == 'male' else negative_prompt_female, 
                                guidance_scale=imgPrompt.guidance_scale, 
                                num_inference_steps=imgPrompt.num_inference_steps, 
                                generator = generator,
                                cross_attention_kwargs={"scale": imgPrompt.strength}
                                ).images[0]

    if not image_30.getbbox():
        image_30: Image = pipe(final_prompt_30,
                                    image=resized_image, strength=imgPrompt.strength + 0.1,
                                    negative_prompt=negative_prompt_male if gender_result.lower() == 'male' else negative_prompt_female,
                                    guidance_scale=imgPrompt.guidance_scale, 
                                    num_inference_steps=imgPrompt.num_inference_steps, 
                                    generator = generator,
                                    cross_attention_kwargs={"scale": imgPrompt.strength}
                                    ).images[0]
        
    # 10 years old generation
    print('Generating 10 years old...')
    image_10: Image = pipe(final_prompt_10,
                                image=resized_image, strength=imgPrompt.strength,
                                negative_prompt=negative_prompt_child, 
                                guidance_scale=imgPrompt.guidance_scale, 
                                num_inference_steps=imgPrompt.num_inference_steps, 
                                generator = generator,
                                cross_attention_kwargs={"scale": imgPrompt.strength}
                                ).images[0]

    if not image_10.getbbox():
        image_10: Image = pipe(final_prompt_10,
                                    image=resized_image, strength=imgPrompt.strength + 0.1,
                                    negative_prompt=negative_prompt_child,
                                    guidance_scale=imgPrompt.guidance_scale, 
                                    num_inference_steps=imgPrompt.num_inference_steps, 
                                    generator = generator,
                                    cross_attention_kwargs={"scale": imgPrompt.strength}
                                    ).images[0]
    
    return image_10, image_30, image_50, image_70
        

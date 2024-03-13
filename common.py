import ast
import logging
from fastchat.model import get_conversation_template
from PIL import Image, UnidentifiedImageError
import requests
import base64
from io import BytesIO
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a tensor with values in [0, 1]
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

prefixes = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

style_prefixes = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

def convert_base64_to_tensor(img_b64_str):
    image_data = base64.b64decode(img_b64_str)
    # Convert binary data to image
    image = Image.open(BytesIO(image_data))
    image = transform(image)
    return image

def convert_base64_to_image(img_b64_str):
    image_data = base64.b64decode(img_b64_str)
    # Convert binary data to image
    image = Image.open(BytesIO(image_data))
    return image

def convert_image_to_base64(image):
    """Given an image, return the base64 encoded image string."""
    try:
        # Load image if it has not been loaded in yet
        if type(image) == str:
            if image.startswith("http://") or image.startswith("https://"):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            elif "base64" in image:
                # OpenAI format is: data:image/jpeg;base64,{base64_encoded_image_str}
                return image.split(",")[1]
            else:
                image = Image.open(image).convert("RGB")

        # max_hw, min_hw = max(image.size), min(image.size)
        # aspect_ratio = max_hw / min_hw
        # max_len, min_len = 2048, 2048
        # shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        # longest_edge = int(shortest_edge * aspect_ratio)
        # W, H = image.size
        # if longest_edge != max(image.size):
        #     if H > W:
        #         H, W = longest_edge, shortest_edge
        #     else:
        #         H, W = shortest_edge, longest_edge
        #     image = image.resize((W, H))

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    except UnidentifiedImageError:
        return None

    return img_b64_str

def load_img(image, trans=None):
    """Given an path, return the image that is normalized to [-1,1]."""
    # if type(image) == str:
    try:
        if image.startswith("http://") or image.startswith("https://"):
            response = requests.get(image)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")

        if trans != None:
            image = trans(image)
    except UnidentifiedImageError:
        return None
        
    return image


def extract_json(s):
    """
    Given an output from the attacker LLM, this function extracts the values
    for `improvement` and `prompt` and returns them as a dictionary.

    Args:
        s (str): The string containing the potential JSON structure.

    Returns:
        dict: A dictionary containing the extracted values.
        str: The cleaned JSON string.
    """
    # Extract the string that looks like a JSON
    start_pos = s.find("{") 
    end_pos = s.find("}") + 1  # +1 to include the closing brace
    if end_pos == -1:
        logging.error("Error extracting potential JSON structure")
        logging.error(f"Input:\n {s}")
        return None, None

    json_str = s[start_pos:end_pos]
    json_str = json_str.replace("\n", "")  # Remove all line breaks

    try:
        parsed = ast.literal_eval(json_str)
        if not all(x in parsed for x in ["improvement","prompt"]):
            logging.error("Error in extracted structure. Missing keys.")
            logging.error(f"Extracted:\n {json_str}")
            return None, None
        return parsed, json_str
    except (SyntaxError, ValueError):
        logging.error("Error parsing extracted structure")
        logging.error(f"Extracted:\n {json_str}")
        return None, None

def get_init_msg():
    return f"**Your objective is to craft a prompt P where when P is entered to a text-to-image generative model, the model generates this exact input image. ** Begin."

def process_init_msg(init_msg, goal_img):
    return [
        {
          "type": "text",
          "text": init_msg,
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{goal_img}",
          },
        },
      ] 

def process_target_response(generated_img, score, goal_img):
    generated_img = convert_image_to_base64(generated_img)
    # goal_img = convert_image_to_base64(goal)
    return [
        {
          "type": "text",
          "text": f"The first image is the GENERATIVE MODEL OUTPUT image and the second image is the OBJECTIVE image. SCORE: {score} ",
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{generated_img}",
          },
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{goal_img}",
          },
        },
      ] 

def conv_template(template_name):
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template
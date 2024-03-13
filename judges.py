from fastchat.model import (
    get_conversation_template
)
import re
from system_prompts import get_judge_system_prompt_img, get_judge_system_prompt_obj, get_judge_system_prompt_style
from language_models import GPT
import lpips
import torch
import open_clip
from common import *

def load_judge(args):
    if args.judge_model == "gpt-4-vision-preview":
        return GPTVJudge(args)
    elif args.judge_model == "no-judge":
        return NoJudge(args)
    else:
        raise NotImplementedError

class JudgeBase:
    def __init__(self, args):
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name = args.judge_model
        # self.goal = args.goal
        if "dreambooth" in args.goal_dir:
            self.system_prompt = get_judge_system_prompt_obj()
        elif "style" in args.goal_dir:
            self.system_prompt = get_judge_system_prompt_style()
        else:
            self.system_prompt = get_judge_system_prompt_img()
        # print(self.system_prompt)

    def get_judge_prompt(self, attack_prompt, target_response):
        return f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"

    def score(self, prompt_list, response_list):
        raise NotImplementedError

    def process_output(self, raw_output):
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            print(f"Error in processing judge output: {raw_output}" )
            output = 1
        return output
               
class NoJudge(JudgeBase):
    def __init__(self, args):
        super(NoJudge, self).__init__(args)
        self.judge_model = None

    def score(self, prompt_list, response_list):
        return [1 for _ in prompt_list]

class GPTVJudge(JudgeBase):
    def __init__(self, args):
        super(GPTVJudge, self).__init__(args)
        self.judge_model = GPT(model_name = self.judge_name)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judge_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()
    
    def get_judge_prompt(self, goal_img, generated_img):
        generated_img = convert_image_to_base64(generated_img)
        # goal_img = convert_image_to_base64(goal)
        return [
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

    def score(self, goal_imgs, target_response_list):
        convs_list = [self.create_conv(self.get_judge_prompt(goal_img, generated_img)) for goal_img, generated_img in zip(goal_imgs, target_response_list)]
        raw_outputs = self.judge_model.batched_generate(convs_list, 
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)
        outputs = [self.process_output(raw_output) for raw_output in raw_outputs]
        return outputs
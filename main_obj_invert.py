import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import argparse
from system_prompts import *
from loggers import WandBLogger
from judges import load_judge
from conversers import load_attack_and_target_models
from common import *
import random
import torch
import time



def main(args):
    args.output_dir = os.path.join(args.output_dir, f"n{args.n_streams}_k{args.n_iterations}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "all_prompts"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "candidates"), exist_ok=True)

    img_dir =  os.path.join(args.goal_dir, args.obj)
    img_paths = [os.path.join(img_dir,filename) for filename in os.listdir(img_dir) if filename.endswith("jpg")]
    
    goal_img_strings = [convert_image_to_base64(img_path) for img_path in img_paths]
    goal_imgs = [load_img(img_path) for img_path in img_paths]

    system_prompt = get_attacker_system_prompt_personalize_obj()
    attackLM, targetLM = load_attack_and_target_models(args)

    judgeLM = load_judge(args)
    
    logger = WandBLogger(args, system_prompt)

    batchsize = args.n_streams
    init_msg = get_init_msg()
    goal_img_idx = random.choices(range(len(goal_img_strings)), k=batchsize)
    goal_img_string_sub = [goal_img_strings[i] for i in goal_img_idx]
    processed_response_list = [process_init_msg(init_msg, goal_img_string_sub[i]) for i in range(batchsize)]
    convs_list = [conv_template(attackLM.template) for _ in range(batchsize)]

    for conv in convs_list:
        conv.set_system_message(system_prompt)
    
    start_time = time.time()
    
    for iteration in range(1, args.n_iterations + 1):
        random.seed(iteration*10)
        print(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")
        goal_img_idx = random.choices(range(len(goal_img_strings)), k=batchsize)
        goal_img_string_sub = [goal_img_strings[i] for i in goal_img_idx]
        
        if iteration > 1:
            processed_response_list = [process_target_response(target_response, score, goal_img_string) for target_response, score, goal_img_string in zip(target_response_list,judge_scores,goal_img_string_sub)]

        extracted_attack_list = attackLM.get_attack(convs_list, processed_response_list)
        print("Finished getting prompts.")

        adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
        improv_list = [attack["improvement"] for attack in extracted_attack_list]
        
        prefix_list = random.choices(prefixes, k=len(adv_prompt_list))
        adv_prompt_list = [prefix.format(adv_prompt) for prefix, adv_prompt in zip (prefix_list, adv_prompt_list)]
                
        if args.target_model == "sdxl-turbo":
            target_response_img_list = targetLM.get_response(adv_prompt_list)
            print("Finished getting target responses.")
            target_response_list = target_response_img_list

        random.shuffle(goal_img_string_sub)
        judge_scores = judgeLM.score(goal_img_string_sub,target_response_list)
        print("Finished getting judge scores.")
        
        for i,(prompt,improv,response, score) in enumerate(zip(adv_prompt_list,improv_list,target_response_list, judge_scores)):
            print(f"{i+1}/{batchsize}\n\n[IMPROVEMENT]:\n{improv} \n\n[PROMPT]:\n{prompt} \n\n[SCORE]:\n{score}\n\n")

        logger.log(iteration, 
                extracted_attack_list,
                target_response_img_list,
                judge_scores)

        for i, conv in enumerate(convs_list):
            conv.messages = conv.messages[-2*(args.keep_last_n):]

    prompt_score_list = logger.finish()
    prompt_score_list = sorted(prompt_score_list, key=lambda x: x[1], reverse=True)
    with open(os.path.join(args.output_dir, f"all_prompts/{args.obj}.txt"), "w") as f:
        for (p,s,i,idx) in prompt_score_list:
            f.write(f"{p},{s},{i},{idx}\n")
    
    candidates = [p for (p,_,_,_) in prompt_score_list[:args.top_c]]
    scores = torch.zeros(len(candidates)).float()
    for g_i in range(len(goal_imgs)):
        target_response_list = targetLM.get_response(candidates)
        scores += torch.tensor(judgeLM.score([goal_img_strings[g_i]]*len(target_response_list),target_response_list))
    scores /= (len(goal_imgs))
    max_score = torch.max(scores)
    max_indices = (scores == max_score).nonzero(as_tuple=True)[0]
    if len(max_indices) == 1:
        best_prompt = candidates[max_indices[0]]
    else:
        best_prompt = sorted([candidates[c] for c in max_indices], key=len)[0]
    
    with open(os.path.join(args.output_dir, f"candidates/{args.obj}.txt"), "w") as f:
        for c in max_indices:
            f.write(f"{candidates[c]},{scores[c]}\n")
    
    with open(os.path.join(args.output_dir, f"{args.obj}.txt"), "w") as f:
        f.write(best_prompt)
    
    print("********* Time", time.time()-start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ########### Attack model parameters ##########
    parser.add_argument(
        "--attack-model",
        default = "gpt-4o-mini",
        help = "Name of attacking model.",
        choices=["gpt-4-vision-preview", "gpt-4o-mini"]
    )
    parser.add_argument(
        "--attack-max-n-tokens",
        type = int,
        default = 500,
        help = "Maximum number of generated tokens for the attacker."
    )
    parser.add_argument(
        "--max-n-attack-attempts",
        type = int,
        default = 5,
        help = "Maximum number of attack generation attempts, in case of generation errors."
    )
    ##################################################

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default = "sdxl-turbo",
        help = "Name of target model.",
        choices=["dall-e-2", "dall-e-3", "sdxl-turbo"]
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type = int,
        default = 150,
        help = "Maximum number of generated tokens for the target."
    )
    ##################################################

    ############ Judge model parameters ##########
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Name of judge model.",
        choices=["gpt-4-vision-preview","no-judge","gpt-4o-mini"]
    )
    parser.add_argument(
        "--judge-max-n-tokens",
        type = int,
        default = 10,
        help = "Maximum number of tokens for the judge."
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0,
        help="Temperature to use for judge."
    )
    ##################################################

    ########### PAIR parameters ##########
    parser.add_argument(
        "--n-streams",
        type = int,
        default = 5,
        help = "Number of concurrent jailbreak conversations. If this is too large, then there may be out of memory errors."
    )

    parser.add_argument(
        "--keep-last-n",
        type = int,
        default = 3,
        help = "Number of responses to save in conversation history of attack model. If this is too large, then it may exceed the context window of the model."
    )
    parser.add_argument(
        "--n-iterations",
        type = int,
        default = 2,
        help = "Number of iterations to run the attack."
    )
    parser.add_argument( 
        "--goal_dir",
        type = str,
        default = "./dreambooth/dataset/",
        help = "directory of the target datasets."
    )
    parser.add_argument( 
        "--obj",
        type = str,
        default = "dog",
        help = "name of the objective"
    )
    parser.add_argument( 
        "--project-name",
        type = str,
        default = "prism_obj_inversion",
        help = "name of the wandb project"
    )
    parser.add_argument(
        "--top-c",
        type = int,
        default = 5,
        help = "Re-evaluate and select the final prompt from the top c candidate prompts"
    )
    parser.add_argument('--english', action='store_true')
    ##################################################

    ########### Logging parameters ##########
    parser.add_argument( 
        "--output-dir",
        type = str,
        default = "obj_results/",
        help = "name of the output directory"
    )

    # TODO: Add a quiet option to suppress print statement
    args = parser.parse_args()

    main(args)

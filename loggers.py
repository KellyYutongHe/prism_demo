import os
import wandb
import pytz
from datetime import datetime
import pandas as pd


class WandBLogger:
    """WandB logger."""

    def __init__(self, args, system_prompt):
        self.logger = wandb.init(
            project = f"{args.project_name}_{args.n_iterations}_{args.n_streams}",
            name = f"{args.obj}",
            config = {
                "attack_model" : args.attack_model,
                "target_model" : args.target_model,
                "judge_model": args.judge_model,
                "keep_last_n": args.keep_last_n,
                "system_prompt": system_prompt,
                "goal_dir": args.goal_dir,
                "n_iter": args.n_iterations,
                "n_streams": args.n_streams,

            }
        )
        self.table = pd.DataFrame()
        self.batch_size = args.n_streams
        self.goal_dir = args.goal_dir
        self.jailbreak_prompt = []
        self.best_score = 0
        self.prompt_score_list = []

    def log(self, iteration: int, attack_list: list, response_list: list, judge_scores: list):
        df = pd.DataFrame(attack_list)
        # df["target_response"] = response_list
        df["judge_scores"] = judge_scores
        df["iter"] = iteration
        df["conv_num"] = [i+1 for i in range(len(response_list))]
        self.table = pd.concat([self.table, df])
        current_best_score = self.table["judge_scores"].max()
        self.prompt_score_list += [(attack_list[i]["prompt"], judge_scores[i], iteration) for i in range(len(attack_list))]
        
        if current_best_score > self.best_score:
            self.best_score = current_best_score
            jailbreak_ind = [i for i in range(len(judge_scores)) if judge_scores[i] == current_best_score]
            self.jailbreak_prompt = [attack_list[i]["prompt"] for i in jailbreak_ind]
            self.jailbreak_response = wandb.Image(response_list[jailbreak_ind[0]])
        elif current_best_score == self.best_score:
            jailbreak_ind = [i for i in range(len(judge_scores)) if judge_scores[i] == current_best_score]
            self.jailbreak_prompt += [attack_list[i]["prompt"] for i in jailbreak_ind]

        self.logger.log({
            "iteration":iteration,
            "judge_scores":judge_scores,
            "generated_images": [wandb.Image(image) for image in response_list],
            "mean_judge_score_iter":sum(judge_scores)/len(judge_scores),
            "max_judge_score":self.table["judge_scores"].max(),
            "min_judge_score":self.table["judge_scores"].min(),
            "jailbreak_prompt":self.jailbreak_prompt,
            "data": wandb.Table(data = self.table)})

        self.print_summary_stats(iteration)

    def finish(self):
        self.print_final_summary_stats()
        self.logger.finish()
        return self.prompt_score_list


    def print_summary_stats(self, iter):
        bs = self.batch_size
        df = self.table 
        mean_score_for_iter = df[df['iter'] == iter]['judge_scores'].mean()
        max_score_for_iter = df[df['iter'] == iter]['judge_scores'].max()
        min_score_for_iter = df[df['iter'] == iter]['judge_scores'].min()

        print(f"{'='*14} SUMMARY STATISTICS {'='*14}")
        print(f"Mean/Max/Min Score for iteration: {mean_score_for_iter}, {max_score_for_iter}, {min_score_for_iter}")

    def print_final_summary_stats(self):
        print(f"{'='*8} FINAL SUMMARY STATISTICS {'='*8}")
        print(f"Goal: {self.goal_dir}")
        df = self.table
        print(f"Best PROMPT:\n\n")
        for prompt in self.jailbreak_prompt:
            print(prompt,"\n")
        print("\n\n")
        max_score = df['judge_scores'].max()
        print(f"Max Score: {max_score}")
        min_score = df['judge_scores'].min()
        print(f"Min Score: {min_score}")
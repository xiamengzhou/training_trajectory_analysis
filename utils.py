import torch
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import math

from transformers import AutoTokenizer, OPTForCausalLM 
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

def return_opt_model_sizes(type="text"):
    """ return the model size of opt models in a textual or numerical format """
    if type == "text":
        return [f"opt_{size}" for size in ["125m", "1.3b", "6.7b", "13b", "30b", "175b"]]
    else:
        return [125*1e6, 1.3*1e9, 6.7*1e9, 13*1e9, 30*1e9, 175*1e9]

def return_checkpoint_index(model_size):
    """ return the checkpoint index (steps) of opt models """
    if model_size == "opt_125m":
        ckpts = sorted(list(range(2000, 40000, 4000)) + list(range(40000, 580000, 20000)))
        ckpts.remove(180000)
        return ckpts
    elif model_size == "opt_1.3b":
        return sorted(list(range(2000, 40000, 4000)) + list(range(40000, 280000, 20000)))
    elif model_size == "opt_6.7b":
        return sorted(list(range(2000, 40000, 4000)) + list(range(40000, 150000, 10000)))
    elif model_size == "opt_13b":
        return sorted(list(range(2000, 72000, 4000)))
    elif model_size == "opt_30b":
        ckpts = sorted(list(range(2000, 72000, 4000)))
        return ckpts
    elif model_size == "opt_175b":
        ckpts = sorted(list(range(40000, 144000, 4000)) + list(range(4000, 40000, 4000)))
        ckpts.remove(28000)
        ckpts.remove(32000)
        ckpts.remove(116000)
        return ckpts
        
def get_num_tokens(model_size, step):
    """ return the trained number of tokens of an opt model at step X """
    if model_size == "opt_125m":
        bs = 0.5 * 1e6
    elif model_size == "opt_1.3b":
        bs = 1 * 1e6
    elif model_size == "opt_6.7b":
        bs = 2 * 1e6
    elif model_size == "opt_13b":
        bs = 4 * 1e6
    elif model_size == "opt_30b":
        bs = 4 * 1e6
    elif model_size == "opt_175b":
        bs = 2 * 1e6
    return bs * step 

def get_flops(model_size, step):
    """ return the trained flops of an opt model at step X """
    if model_size == "opt_125m":
        m = 125 * 1e6
    elif model_size == "opt_1.3b":
        m = 1.3 * 1e9
    elif model_size == "opt_6.7b":
        m = 6.7 * 1e9
    elif model_size == "opt_13b":
        m = 13 * 1e9
    elif model_size == "opt_30b":
        m = 30 * 1e9
    elif model_size == "opt_175b":
        m = 175 * 1e9
    t = get_num_tokens(model_size, step)
    return 2 * 3 * m * t

def return_color(model_size="opt_125m"):
    """ return the color of opt models """
    colors = [plt.cm.tab20(i) for i in range(20)]
    if model_size == "opt_125m":
        return colors[0]
    elif model_size == "opt_1.3b":
        return colors[2]
    elif model_size == "opt_6.7b":
        return colors[4]
    elif model_size == "opt_13b":
        return colors[6]
    elif model_size == "opt_30b":
        return colors[8]
    elif model_size == "opt_175b":
        return colors[10]
    else:
        return colors[14]
    
    
def load_opt_validation_ppl():
    """ 
        load the perplexity trajectory of opt models 
        level-1 key: opt_125m, opt_1.3b, opt_6.7b, opt_13b, opt_30b, opt_175b
        level-2 key: valid/combined and perplexity of each set
        level-3 key: steps
    """
    return torch.load("data/opt_validation_ppl_trajectory.pt")


def read_jsonl(file):
    """ read jsonl file """
    ds = []
    try:
        with open(file) as f:
            for i, line in enumerate(f):
                d = json.loads(line.strip())
                ds.append(d)
    except:
        print("Error reading file:", file)
        return 
    return ds


def collect_trend_of_tokens(dataset_name="gutenberg_pg-19", threshold=0.1):
    """ collect the trend of tokens from checkpoints and save it """
    
    # format: {model_size: np.array(num_tokens, num_checkpoints)}
    ppl_scores = torch.load(f"data/trend_of_tokens/all_ppls-{dataset_name}.pt")

    def linear_with_statistical_test(all_scores, threshold, model_size):
        # collect the trend of tokens from checkpoints starting from threshold (%) of training to the end
        # make sure that the checkpoints are evenly spaced -- otherwise you would have to rejust the threshold
        checkpoint_index = np.array(return_checkpoint_index(model_size))
        threshold_ckpt_index = int(checkpoint_index[-1] * threshold)
        starting_index = np.argmin(np.abs(checkpoint_index - threshold_ckpt_index))

        all_scores = all_scores[:, starting_index:]
        pvalues = np.full([all_scores.shape[0], 2], np.nan)
        params = np.full([all_scores.shape[0], 2], np.nan)
        
        for i, y in enumerate(all_scores):
            y = y / y[0]
            y = y[~np.isnan(y)]
            if len(y) == 0:
                continue
            x = np.log(checkpoint_index[starting_index])
            x = sm.add_constant(x)
            y = y.reshape(-1)
            reg = sm.OLS(y, x).fit()
            pvalues[i] = reg.pvalues
            params[i] = reg.params
            if i % 10000 == 0:
                print(i, flush=True)
        return pvalues, params 
    
    for model_size in ppl_scores:
        output_file = f"data/trend_of_tokens/linear_trend/slo-{dataset_name}-{model_size}-simple.pt"
        coefs = {}
        for threshold in [0.1, 0.3, 0.7]: 
            ppls = ppl_scores 
            pvalues, params  = linear_with_statistical_test(ppls, threshold)
            coefs[threshold] = {"pvalues": pvalues, "params": params}
            print(f"Finished model size: {model_size}", flush=True)
        torch.save(coefs, output_file)
    
def pull_out_generation(data_file, n=5, seed=0):
    """ randomly sample an example from a jsonl file that contains generated sequences in entry "text" """
    data = read_jsonl(data_file)
    data = [d["text"] for d in data]
    np.random.seed(seed)
    selection = np.random.choice(range(len(data)), n, replace=False)
    print("Selected index:", selection)
    for i in selection:
        text = data[i]
        print("*"*30)
        print(text)

def get_xaxis(model_size, steps, xaxis_type, xaxis_log=True):
    """ return the x-axis of opt models 
        xaxis_type: Steps, FLOPs, Tokens, ppl
    """
    valid_ppls = load_opt_validation_ppl()
    if xaxis_type == "Steps":
        x = steps
    elif xaxis_type == "FLOPs":
        x = [get_flops(model_size, s) for s in steps]
    elif xaxis_type == "Tokens":
        x = [get_num_tokens(model_size, s) for s in steps]
    elif xaxis_type == "ppl":
        x = [valid_ppls[model_size]["valid/combined"][s] for s in steps]
    if xaxis_log: x = np.log10(x)
    return x

def bigbench_name_matching_dict():
    """ return a dictionary that maps `bigbench__{taskname}` to `{task_name}` """
    analysis = torch.load(f"data/bigbench/dataset_analysis.pt")
    d = {f"bigbench__{task.replace('_', '')}": task for task in analysis} 
    return d

def get_random_performance_bigbench(task):
    """ return the random performance of a bigbench task 
        Input:
            task: bigbench__{taskname} or {task_name}
    """
    def calculate_random_acc(task_analysis):
        accu = []
        for choice_num in task_analysis["choice_num"]:
            occur = task_analysis["choice_num"][choice_num]
            accu.extend([1/choice_num] * occur)
        return np.mean(accu) * 100
    
    analysis = torch.load(f"data/bigbench/dataset_analysis.pt")
    task_name_m = bigbench_name_matching_dict() 
    if task.startswith("bigbench__"):
        task = task_name_m[task]

    if task == "periodic_elements":
        return 1 / 118
    return calculate_random_acc(analysis[task])

def get_bigbench_tasks():
    """ select bigbench tasks that satisfy the following conditions: have less than 1000 examples and constant number of options """ 
    analysis = torch.load(f"data/bigbench/dataset_analysis.pt")
    count = 0
    task_set = []

    def cond1(task):
        return analysis[task]["num"] < 1000 and set(analysis[task]["target_num"].keys()) == set([1])

    for task in analysis:
        if len(analysis[task]) > 0:
            if cond1(task):
                count += 1
                task_set.append(task)
                
    task_set = [f"bigbench__{t.replace('_', '')}" for t in task_set]
    task_set += ["bigbench__periodicelements"]
    return task_set

def load_bigbench_random_performance(task):
    """ return the random performance of a bigbench task """
    def calculate_random_acc(task_analysis):
        accu = []
        for choice_num in task_analysis["choice_num"]:
            occur = task_analysis["choice_num"][choice_num]
            accu.extend([1/choice_num] * occur)
        return np.mean(accu) * 100
    analysis = torch.load(f"data/bigbench/dataset_analysis.pt")
    task_name_m = bigbench_name_matching_dict() 
    if task.startswith("bigbench__"):
        task = task_name_m[task]

    if task == "periodic_elements":
        return 1 / 118
    return calculate_random_acc(analysis[task])


def load_correct_incorrect_option_ppl():
    """ return the perplexity of correct and incorrect examples 
        format: {model_size: {"correct_ppl": {step:}, "incorrect_ppl": {step:}, "correct_token_num: {step:}, "incorrect_token_num": {step:}}}
    """
    ppls = {}
    for model_size in return_opt_model_sizes():
        out_dir = f"data/bigbench/perplexity_analysis"
        out_file = os.path.join(out_dir, f"{model_size}.pt")
        ppls[model_size] = torch.load(out_file)[model_size]
    return ppls

def reorganize_option_ppl(tasks):
    """ reorganize option perplexity results into the following format:
        {model: np.array(num_tasks, num_steps)}
    """
    correct_and_incorrect_ppls = load_correct_incorrect_option_ppl()
    all_correct_ppls, all_incorrect_ppls, all_correct_token_nums, all_incorrect_token_nums = {}, {}, {}, {}
    ckpts = {}
    num_tasks = len(tasks)
   
    for model_size in correct_and_incorrect_ppls:
        ckpts = return_checkpoint_index(model_size)
        incorrect_ppls_tasks = np.full((num_tasks, len(ckpts)), -1.)
        correct_ppls_tasks = np.full((num_tasks, len(ckpts)), -1.)
        incorrect_token_num_tasks = np.full(num_tasks, -1.)
        correct_token_num_tasks = np.full(num_tasks, -1.)
        
        correct_ppl = correct_and_incorrect_ppls[model_size]["correct_ppl"]
        incorrect_ppl = correct_and_incorrect_ppls[model_size]["incorrect_ppl"]
        correct_token_num = correct_and_incorrect_ppls[model_size]["correct_token_num"]
        incorrect_token_num = correct_and_incorrect_ppls[model_size]["incorrect_token_num"]

        for i, task in enumerate(tasks):
            if task not in correct_ppl: continue
            for j, ckpt in enumerate(ckpts):
                if task in incorrect_token_num:
                    incorrect_token_num_tasks[i] = incorrect_token_num[task]
                if task in correct_token_num:
                    correct_token_num_tasks[i] = correct_token_num[task]
                if ckpt in correct_ppl[task]:
                    correct_ppls_tasks[i][j] = correct_ppl[task][ckpt]
                    incorrect_ppls_tasks[i][j] = incorrect_ppl[task][ckpt]
                    
        all_correct_ppls[model_size] = correct_ppls_tasks
        all_incorrect_ppls[model_size] = incorrect_ppls_tasks
        all_correct_token_nums[model_size] = correct_token_num_tasks
        all_incorrect_token_nums[model_size] = incorrect_token_num_tasks

    def aggregate_option_perplexity(ppl_tokens, num_tokens):
        # ppl_tokens: # tasks * # ckpts
        # num_tokens: # tasks
        return np.exp(np.sum(np.log(ppl_tokens) * num_tokens.reshape(-1, 1), axis=0) / np.sum(num_tokens))
    
    aggregated_correct_option_ppls = {}; aggregated_incorrect_option_ppls = {}
    for model_size in correct_and_incorrect_ppls:
        aggregated_correct_option_ppls[model_size] = aggregate_option_perplexity(all_correct_ppls[model_size], all_correct_token_nums[model_size])
        aggregated_incorrect_option_ppls[model_size] = aggregate_option_perplexity(all_incorrect_ppls[model_size], all_incorrect_token_nums[model_size])
    return all_correct_ppls, aggregated_correct_option_ppls, aggregated_incorrect_option_ppls

     
def exponential_moving_average(ys, weight=0.1):
    """ return the exponential moving average of ys """
    ema_ys = []
    for i in range(len(ys)):
        if i == 0:
            ema_ys.append(ys[0])
        else:
            ema_ys.append((ema_ys[-1] * weight + ys[i] * (1-weight)))
    return ema_ys


class OPTSubtract(OPTForCausalLM):
    def __init__(self, config, small_lm, weight1=1, weight2=-1):
        super().__init__(config)
        self.small_lm = OPTForCausalLM.from_pretrained(small_lm)
        self.weight1 = weight1
        self.weight2 = weight2
    
    def forward(self, **kwargs):
        """
        kwargs will include
        - input_ids
        - attention_mask
        - past_key_values: (large model, small model)
        - use cache
        - return_dict
        - output_attentions
        - output_hidden_states

        The small model should share all of them except past_key_values.
        """
        large_model_input = kwargs.copy()
        small_model_input = kwargs.copy()
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
            large_model_input['past_key_values'] = kwargs['past_key_values'][0]
            small_model_input['past_key_values'] = kwargs['past_key_values'][1]

        large_model_output = super().forward(**large_model_input)
        small_model_output = self.small_lm(**small_model_input)

        subtract_prob = self.weight1 * F.softmax(large_model_output.logits, -1) + self.weight2 * F.softmax(small_model_output.logits, -1)
        
        subtract_prob[subtract_prob < 0] = 0
        subtract_prob = subtract_prob + 1e-7
        new_logits = subtract_prob.log() # No need to normalize because this is the logit

        return CausalLMOutputWithPast(
            loss=(large_model_output.loss, small_model_output.loss),
            logits=new_logits,
            past_key_values=None, # (large_model_output.past_key_values, small_model_output.past_key_values),
            hidden_states=(large_model_output.hidden_states, small_model_output.hidden_states),
            attentions=(large_model_output.attentions, small_model_output.attentions),
        )
        
def decode(model1_name="facebook/opt-350m", model2_name="facebook/opt-125m", weight1=1, weight2=-1, sampling=False, device="cuda"):
    model = OPTSubtract.from_pretrained(model1_name, small_lm=model2_name, weight1=weight1, weight2=weight2)
    model.small_lm = OPTForCausalLM.from_pretrained(model2_name)
    model.weight1 = weight1
    model.weight2 = weight2
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model1_name)
    prompt = "The weather is nice today. Let's"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # neucleus sampling (sampling=True): </s>The weather is nice today. Let's go to the park for some sun and picture taking!</s> 
    # greedy search (sampling=False): </s>The weather is nice today. Let's go to the park and play some games!</s>
    generations = model.generate(input_ids, do_sample=sampling, max_new_tokens=20, top_k=None, top_p=0.95)
    print(tokenizer.decode(generations[0]))
    
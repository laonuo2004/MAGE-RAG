import os
import sys

if sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import json
import re
import argparse
import time
from typing import Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
import numpy as np
import tiktoken
from openai import OpenAI
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import evaluate_prompt

LLM_BASE_URL = "<YOUR_BASE_URL>"
LLM_API_KEY = "<YOUR_API_KEY>"


PRICE_INPUT = 0.5
PRICE_OUTPUT = 1.5


class Evaluator:
    def __init__(self, model: str = "gpt-4o-mini", error_log_file: str = None):
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.evaluate_prompt_template = evaluate_prompt
        
        self.error_log_file = error_log_file
        self.error_lock = threading.Lock()
    
    def extract_model_answer(self, model_answer: str) -> str:
        output = re.findall(r'<answer>(.*?)</answer>', model_answer, re.DOTALL)
        if len(output) == 0:
            return None
        return output[0].strip()
    
    def query_llm(self, prompt, model, tokenizer, client=None, temperature=0.0, max_new_tokens=128, stop=None, images: List[str] = None):
        # truncate
        max_len = 4096  # Default max length
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
        tries = 0
        model_name = model
        while tries < 5:
            tries += 1
            try:
                if images and images:
                    # Vision-enabled prompt: include images as base64
                    content = [
                        {"type": "text", "text": prompt}
                    ]
                    for img_base64 in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        })
                    messages = [{"role": "user", "content": content}]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                return completion.choices[0].message.content
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(f"Error Occurs: \"{str(e)}\"        Retry ...")
                time.sleep(1)
        else:
            print("Max tries. Failed.")
            return ''

    def evaluate_correctness(self, pred: str, answer: str, question: str) -> Dict[str, Any]:
        """Evaluate prediction correctness using LLM"""
        if pred is None or answer is None or question is None:
            return {"accuracy": 0, "reasoning": "Input parameters are empty"}
    
        prompt = self.evaluate_prompt_template.format(question=question, gold_answers=answer, assistant_answer=pred)
        attempt = 0
        while attempt < 2:
            attempt += 1
            response = self.query_llm(prompt, self.model, self.tokenizer, self.client, temperature=0.0, max_new_tokens=300)
            eval_result = None

            # First try direct JSON parse
            try:
                eval_result = json.loads(response)
            except json.JSONDecodeError:
                # Fallback: extract JSON-like substring containing accuracy
                json_match = re.search(r'\{[^}]*"accuracy"\s*:\s*(\d+)[^}]*\}', response, re.DOTALL)
                if json_match:
                    try:
                        eval_result = json.loads(json_match.group(0))
                    except Exception:
                        eval_result = None
                else:
                    eval_result = None
            except Exception:
                eval_result = None

            if eval_result is not None:
                try:
                    accuracy = int(eval_result.get('accuracy', 0))
                except Exception:
                    accuracy = 0
                reasoning = eval_result.get('reasoning', '')
                return {"accuracy": accuracy, "reasoning": reasoning}

            print(f"Invalid evaluate response (attempt {attempt}), retrying...")
            time.sleep(1)
        # After max attempts
        return {"accuracy": 0, "reasoning": f"json解析失败，输出为{response}"}

    def evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        q_id = sample.get('_id', 'unknown')
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        model_answer = sample.get('response', '')
        times = sample.get('process_time', '')
        input_tokens = sample.get('usage', {}).get('prompt_tokens', 0)
        output_tokens = sample.get('usage', {}).get('completion_tokens', 0)

        new_sample = {
            'q_id': q_id,
            'question': question,
            'answer': answer,
            'model_answer': model_answer,
            'times': times,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }

        extracted_answer = self.extract_model_answer(model_answer)
        try:
            eval_result = self.evaluate_correctness(extracted_answer, answer, question)
            new_sample['eval_accuracy'] = eval_result['accuracy']
            new_sample['eval_reasoning'] = eval_result['reasoning']
            new_sample['extracted_answer'] = extracted_answer
            
            return {
                "sample": new_sample,
                "status": "evaluated",
                "accuracy": eval_result['accuracy']
            }
            
        except Exception as e:
            print(f"\n⚠️ evaluate sample {q_id} failed: {e}")
            new_sample['eval_accuracy'] = 0
            new_sample['eval_reasoning'] = f"evaluate failed: {str(e)}"
            new_sample['extracted_answer'] = model_answer
            
            return {
                "sample": new_sample,
                "status": "error",
                "accuracy": 0
            }
    
    def evaluate_file(self, input_file: str, output_file: str = None, num_threads: int = 4, error_log_file: str = None) -> Dict[str, Any]:
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_evaluated{input_path.suffix}")
        
        print(f"📖 read input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f if line.strip()]
        print(f"✅ total {len(samples)} samples")
        print(f"🚀 use {num_threads} threads to evaluate")
        
        stats = {
            "total": len(samples),
            "evaluated": 0,
            "skipped": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy_rate": 0.0,
            "input_token": 0,
            "output_token": 0
        }
        
        print(f"🔍 start evaluating...")
        evaluated_samples = [None] * len(samples)
        lock = threading.Lock()
        
        if num_threads == 1:
            for idx, sample in enumerate(tqdm(samples, desc="evaluating progress")):
                result = self.evaluate_single_sample(sample)
                evaluated_samples[idx] = result["sample"]
                
                with lock:
                    if result["status"] == "evaluated":
                        stats["evaluated"] += 1
                    elif result["status"] == "skipped":
                        stats["skipped"] += 1
                        
                    if result["accuracy"] == 1:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_idx = {
                    executor.submit(self.evaluate_single_sample, sample): idx 
                    for idx, sample in enumerate(samples)
                }
                
                with tqdm(total=len(samples), desc="evaluating progress") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            evaluated_samples[idx] = result["sample"]
                            
                            with lock:
                                if result["status"] == "evaluated":
                                    stats["evaluated"] += 1
                                elif result["status"] == "skipped":
                                    stats["skipped"] += 1
                                
                                if result["accuracy"] == 1:
                                    stats["correct"] += 1
                                else:
                                    stats["incorrect"] += 1
                            
                        except Exception as e:
                            print(f"\n⚠️ process sample {idx} failed: {e}")
                            evaluated_samples[idx] = samples[idx]
                            evaluated_samples[idx]['eval_accuracy'] = 0
                            evaluated_samples[idx]['eval_reasoning'] = f"process failed: {str(e)}"
                            with lock:
                                stats["incorrect"] += 1
                    
                        pbar.update(1)
        
        total_evaluated = stats["correct"] + stats["incorrect"]
        stats["input_token"] = np.mean([s['usage']["prompt_tokens"] for s in samples])
        stats["output_token"] = np.mean([s['usage']["completion_tokens"] for s in samples])

        if total_evaluated > 0:
            stats["accuracy_rate"] = stats["correct"] / total_evaluated
        
        print(f"💾 save evaluation result to: {output_file}")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in evaluated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print("\n" + "="*60)
        print("📊 evaluation statistics:")
        print(f"  total samples: {stats['total']}")
        print(f"  evaluated samples: {stats['evaluated']}")
        print(f"  skipped samples: {stats['skipped']}")
        print(f"  correct number: {stats['correct']}")
        print(f"  incorrect number: {stats['incorrect']}")
        print(f"  accuracy rate: {stats['accuracy_rate']:.2%}")
        print("="*60)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="evaluate the model generated answers")
    parser.add_argument('--input_file', type=str, default="")
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--num_threads', type=int, default=50)
    parser.add_argument('--num_evals', type=int, default=2, help='number of evaluations for each file')
    parser.add_argument('--summary_file', type=str, default=None, help='summary result file path')
    
    args = parser.parse_args()

    input_files=[] 
 
    base_paths = [
        '<YOUR_MODEL_NAME>',
    ]
    domain=['slidevqa','scgqa','spiqa','feta_tab','paper_tab']

    # group files by base path    
    files_by_base_path = {}
    for base_path in base_paths:
        files_by_base_path[base_path] = []
        for d in domain:
            input_file = f"results/{d}/{base_path}_dag_rag_1.jsonl"
            files_by_base_path[base_path].append(input_file)

    evaluator = Evaluator(model=args.model)
    
    # evaluate each base path separately
    for base_path, input_files in files_by_base_path.items():
        print(f"\n{'#'*80}")
        print(f"🚀 start evaluating base path: {base_path}")
        print(f"{'#'*80}\n")
        
        # generate independent summary file for each base path
        if args.summary_file is None:
            first_output = input_files[0].replace('results', 'eval_result/DAG')
            base_eval_dir = os.path.dirname(os.path.dirname(first_output))
            current_summary_file = os.path.join(base_eval_dir, 'evaluation_summary.jsonl')
        else:
            summary_dir = os.path.dirname(args.summary_file)
            summary_name = os.path.basename(args.summary_file)
            name_parts = os.path.splitext(summary_name)
            current_summary_file = os.path.join(summary_dir, f"{name_parts[0]}_{base_path}{name_parts[1]}")
        
        summary_dir = os.path.dirname(current_summary_file)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(current_summary_file, 'w', encoding='utf-8') as f:
            pass
        print(f"📝 the summary result of the current base path will be saved to: {current_summary_file}\n")
        
        all_results = []
        input_tokens = []
        output_tokens = []

        for input_file in input_files:
            print(f"\n{'='*80}")
            print(f"📁 evaluate file: {input_file}")
            print(f"{'='*80}")
                    
            accuracy_rates = []
            output_file = input_file.replace('results', 'eval_result/DAG')
            # output_file = args.output_file
            if input_file == output_file:
                raise ValueError(f"input file and output file are the same: {input_file} == {output_file}")
            for eval_round in range(args.num_evals):
                print(f"\n🔄 the {eval_round + 1}/{args.num_evals} evaluation...")
                stats = evaluator.evaluate_file(
                    input_file=input_file,
                    output_file=output_file,
                    num_threads=args.num_threads
                )
                
                accuracy_rates.append(stats['accuracy_rate'])
                print(f"    accuracy rate: {stats['accuracy_rate']:.4f}")
            
            input_tokens.append(stats['input_token'])
            output_tokens.append(stats['output_token'])
            
            mean_accuracy = np.mean(accuracy_rates)
            std_accuracy = np.std(accuracy_rates, ddof=1) 
            mean_input_token = np.mean(input_tokens)
            mean_output_token = np.mean(output_tokens)
            
            file_result = {
                'file': input_file,
                'output_file': output_file,
                'num_samples': stats['total'],
                'num_evaluations': args.num_evals,
                'accuracy_rates': accuracy_rates,
                'mean_accuracy': float(mean_accuracy),
                'std_accuracy': float(std_accuracy),
                'timestamp': datetime.now().isoformat(),
                'input_token': float(mean_input_token),
                'output_token': float(mean_output_token),
            }
            
            all_results.append(file_result)
            
            print(f"\n📊 summary statistics:")
            print(f"    average accuracy rate: {mean_accuracy:.4f}")
            print(f"    standard deviation: {std_accuracy:.4f}")
            print(f"    accuracy rates for each round: {[f'{acc:.4f}' for acc in accuracy_rates]}")
        
            # immediately append to summary file
            print(f"💾 append to summary file: {current_summary_file}")
            with open(current_summary_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(file_result, ensure_ascii=False) + '\n')
        
        # after the evaluation of the current base path, output the summary
        print(f"\n{'='*80}")
        print(f"✅ base path {base_path} evaluation completed!")
        print(f"{'='*80}")
        print(f"total files: {len(all_results)}")
        print(f"\ndomain: {domain} detailed results:")
        for result in all_results:
            filename = os.path.basename(os.path.dirname(result['output_file']))
            print(f"  {filename}:")
            print(f"     average accuracy rate: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
        
        # calculate total token count and total cost
        total_cost = (mean_input_token / 1000000 * PRICE_INPUT) + (mean_output_token / 1000000 * PRICE_OUTPUT)
        
        print(f"the overall average result is: {np.mean([result['mean_accuracy'] for result in all_results]):.4f} ± {np.mean([result['std_accuracy'] for result in all_results]):.2f}")
        print(f"average input token per file: {mean_input_token:.0f}; average output token per file: {mean_output_token:.0f}")
        print(f"average total token consumption: {mean_input_token + mean_output_token:.0f}")
        print(f"average cost: ${total_cost:.4f} USD")
        print(f"{'='*80}")

        # save to txt file
        txt_fname = os.path.join(os.path.dirname(os.path.dirname(input_files[0])), 'results.txt')
        with open(txt_fname, 'w', encoding='utf-8') as f:
            for result in all_results:
                filename = os.path.basename(os.path.dirname(result['file']))
                f.write(f"{filename}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}\n")
            f.write(f"Average: {np.mean([result['mean_accuracy'] for result in all_results]):.4f} ± {np.mean([result['std_accuracy'] for result in all_results]):.4f}\n")
            f.write(f"Total Input Tokens: {mean_input_token:.0f}\n")
            f.write(f"Total Output Tokens: {mean_output_token:.0f}\n")
            f.write(f"Total Cost: ${total_cost:.4f} USD\n")
    
    print(f"\n{'#'*80}")
    print("🎉 all base paths evaluation completed!")
    print(f"{'#'*80}")
if __name__ == "__main__":
    main()


import argparse
from agent_search.pred_kw import DAGPred

if __name__ == "__main__":
    print("Starting execution")
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--n_proc", "-n", type=int, default=30)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--method", type=str, default="dag", 
                        choices=["dag"])
    parser.add_argument("--rag", type=int, default=1)
    parser.add_argument("--self_refine", default=False, action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--judge_use", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--top_k_text_kw", type=int, default=3)
    parser.add_argument("--top_k_image_kw", type=int, default=3)
    parser.add_argument("--keyword_modality", default = "all")
    parser.add_argument("--keyword", default=True)
    parser.add_argument("--use_dag", action="store_true", help="Use DAG mode")
    args = parser.parse_args()

    method = DAGPred(args)
    method.main()
    
    print("All tasks completed")
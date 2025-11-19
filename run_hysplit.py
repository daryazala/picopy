import automate_hysplit as ah

import multiprocessing as mp
import os
import subprocess

def run_on_gpu(gpu_id, task_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running task {task_id} on GPU {gpu_id}")
    # Example: call your GPU code (e.g., Python script, PyTorch model, etc.)
    subprocess.run(["python", "your_gpu_script.py", f"--task={task_id}"])

if __name__ == "__main__":
    n_gpus = 4
    processes = []
    for i in range(n_gpus):
        p = mp.Process(target=run_on_gpu, args=(i, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


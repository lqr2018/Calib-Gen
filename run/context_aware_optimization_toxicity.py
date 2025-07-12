import subprocess
from concurrent.futures import ThreadPoolExecutor
from utils.utils import make_filename

class CommandExecutor:
    def __init__(self, base_command, variations1, variations2, max_concurrent_processes=1):
        self.base_command = base_command
        self.variations1 = variations1
        self.variations2 = variations2
        self.commands = []
        self.processes = []

        self.max_concurrent_processes = max_concurrent_processes

        self._generate_commands()

    def _generate_commands(self):
        for variation1 in self.variations1:
            for variation2 in self.variations2:
                command = self.base_command + [
                    "--prompts_setting", str(variation1["prompts_setting"]),
                    "--output_prompts_setting", str(variation1["prompts_setting"]+'_cao'),
                    "--strength", str(variation2["strength"]),
                ]
                self.commands.append(command)


    def _run_command(self, command, i):
        try:
            print(f"Executing command {i + 1}/{len(self.commands)}: {' '.join(command)}")
            process = subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            self.processes.append(process)

            for line in process.stdout:
                print(line, end="")
            for error in process.stderr:
                print(f"ERROR: {error}", end="")

            process.wait()
            print(f"Command {i + 1} completed.\n")
        except subprocess.CalledProcessError as e:
            print("Command execution failed")
            print("Exit status code:", e.returncode)
            print("Error output:\n", e.stderr)

    def start_commands(self):
        with ThreadPoolExecutor(max_workers=self.max_concurrent_processes) as executor:
            futures = [executor.submit(self._run_command, command, i) for i, command in enumerate(self.commands)]
            for future in futures:
                future.result()


if __name__ == "__main__":
    base_command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python", "scripts/data_generation/toxicity_context_aware_optimization.py",
    ]

    save_dir = "evaluation_outputs"
    prefix_setting = "pos"
    strength = -1.0
    top_k = 100
    top_p = 0.8
    gamma = 0.0
    thresh = 0
    method = "calib"

    variations1 = [
        # {"prompts_setting": "my_toxicity_random_small"},
        # {"prompts_setting": "my_toxicity_random"},

        # {"prompts_setting": "my_toxicity_toxic_small"},
        # {"prompts_setting": "my_toxicity_toxic"},
    ]

    variations2 = [
        # {"strength":  "0.0"},
        # {"strength": "-0.1"},
        {"strength": "-0.2"},
        # {"strength": "-0.5"},
        # {"strength": "-1.0"},
        # {"strength": "-2.0"},
        # {"strength": "-5.0"},
    ]


    executor = CommandExecutor(base_command, variations1, variations2, max_concurrent_processes=1)

    try:
        executor.start_commands()
    except Exception as e:
        print(f"Unhandled error occurred: {e}")

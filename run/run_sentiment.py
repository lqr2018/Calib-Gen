import subprocess
from concurrent.futures import ThreadPoolExecutor


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
                # print(variation2)
                command = self.base_command + [
                    "--method", str(variation1["method"]),
                    "--prompts_setting", str(variation2["prompts_setting"]),
                    "--prefix_setting", str(variation2["prefix_setting"]),
                ]
                self.commands.append(command)

    def _run_command(self, command, i):
        try:
            print(f"Executing command {i + 1}/{len(self.commands)}: {' '.join(command)}")
            process = subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        "python", "scripts/experiments/evaluate_sentiment.py",
        "--save_dir", "evaluation_outputs",
        "--strength", "2.0",
        "--base_model_string", "THUDM/chatglm3-6b",
        "--top_k", "0",
        "--top_p", "0.8",
    ]

    variations1 = [
        # {"method": "preadd"},
        {"method": "calib"},
        # {"method": "fudge"},
        # {"method": "neg_prompting"},
        # {"method": "raw_prompting"},
    ]

    variations2 = [
        # {"prompts_setting": "sentiment_positive", "prefix_setting":  "neg"},
        # {"prompts_setting": "sentiment_negative", "prefix_setting":  "pos"},
        {"prompts_setting": "sentiment_positive_cao", "prefix_setting": "neg"},
        # {"prompts_setting": "sentiment_negative_cao", "prefix_setting": "pos"},
    ]

    executor = CommandExecutor(base_command, variations1, variations2, max_concurrent_processes=1)

    try:
        executor.start_commands()
    except Exception as e:
        print(f"Unhandled error occurred: {e}")

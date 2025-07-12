import os
import subprocess
from utils.utils import make_filename

if __name__ == '__main__':
    commands = []

    save_dir = "evaluation_outputs"
    prefix_setting = "pos"
    strength = -1.0
    top_k = 0
    top_p = 0.8

    variations1 = [
        # {"method": "preadd"},
        {"method": "calib"},
        # {"method": "fudge"},
        # {"method": "neg_prompting"},
        # {"method": "raw_prompting"},
    ]

    variations2 = [
        # {"prompts_setting": "toxicity_random_small"},
        {"prompts_setting": "toxicity_random"},

        # {"prompts_setting": "toxicity_random_small_cao"},
        # {"prompts_setting": "toxicity_random_cao"},

        # {"prompts_setting": "toxicity_toxic_small"},·
        {"prompts_setting": "toxicity_toxic"},

        # {"prompts_setting": "toxicity_toxic_small_cao"},
        # {"prompts_setting": "toxicity_toxic_cao"},
    ]

    for variation1 in variations1:
        for variation2 in variations2:
            filename = make_filename(save_dir=save_dir,
                                     prompts_setting=variation2["prompts_setting"],
                                     method=variation1["method"],
                                     prefix_setting=prefix_setting,
                                     strength=strength,
                                     top_k=top_k,
                                     top_p=top_p,
                                     )

            command = [
                "python", "scripts/analysis/analyze_toxicity_results.py",
                "--outputs", filename
            ]

            commands.append(command)

    # 使用 subprocess 启动命令
    for i, command in enumerate(commands):
        print(f"Executing command {i + 1}/{len(commands)}: {' '.join(command)}")
        try:
            result = subprocess.run(
                " ".join(command),
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print("Command executed successfully")
            print("Standard output:\n", result.stdout)
            print("Standard error:\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Command execution failed")
            print("Exit code:", e.returncode)
            print("Error output:\n", e.stderr)

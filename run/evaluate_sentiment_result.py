import os
import subprocess
from utils.utils import make_filename

if __name__ == '__main__':
    commands = []

    save_dir = "evaluation_outputs"
    strength = 2.0
    top_k = 0
    top_p = 0.8
    gamma = 0.0
    thresh = 0


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

    for variation1 in variations1:
        for variation2 in variations2:
            filename = make_filename(save_dir=save_dir,
                                     prompts_setting=variation2["prompts_setting"],
                                     method=variation1["method"],
                                     prefix_setting=variation2["prefix_setting"],
                                     strength=strength,
                                     gamma=gamma,
                                     top_k=top_k,
                                     top_p=top_p,
                                     thresh=thresh
                                     )

            command = [
                "python", "scripts/analysis/analyze_sentiment_results.py",
                "--outputs", filename
            ]

            commands.append(command)

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

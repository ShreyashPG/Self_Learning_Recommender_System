import subprocess

scripts = [
    "data_preparation_context.py",
    "contextual_bandit.py",
    "app_context.py",
    "app_ab_testing.py",
    # Assume user interacts with the app here before analysis
    "analyze_ab_testing.py",
    "data_preparation_context.py",
    "ncf_model.py",
    "contextual_bandit.py",
    "app_ab_testing.py",
    "analyze_ab_testing.py"
]

def run_scripts(scripts):
    for script in scripts:
        print(f"\nRunning: {script}")
        result = subprocess.run(["python", script], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)

if __name__ == "__main__":
    run_scripts(scripts)

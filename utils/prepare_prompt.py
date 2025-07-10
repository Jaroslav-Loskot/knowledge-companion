import os

# Folders to ignore even in the root directory
IGNORED_DIRS = {'.venv', 'venv', '.git', '__pycache__', '.mypy_cache', '.idea', '.vscode'}

def load_python_files(base_path):
    output = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)

        # Skip ignored directories and non-.py files
        if item in IGNORED_DIRS or not item.endswith(".py") or not os.path.isfile(item_path):
            continue

        try:
            with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                output.append(f"{'-'*80}\n{item_path}\n{'-'*80}\n{content}\n")
        except Exception as e:
            output.append(f"{'-'*80}\n{item_path}\n{'-'*80}\n[Error reading file: {e}]\n")

    return "\n".join(output)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    result = load_python_files(parent_dir)
    print(result)

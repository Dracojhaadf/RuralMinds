"""
Fixes tokenizer_config.json in fine-tuned Malayalam model directory.
"""
import json
import shutil
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import MALAYALAM_MODEL_PATH

CONFIG_FILE = Path(MALAYALAM_MODEL_PATH) / "tokenizer_config.json"

def main():
    if not CONFIG_FILE.exists():
        print(f"File not found: {CONFIG_FILE}")
        return

    backup = CONFIG_FILE.with_suffix(".json.bak")
    shutil.copy(CONFIG_FILE, backup)
    print(f"✅ Backup saved to: {backup}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    val = cfg.get("extra_special_tokens")
    print(f"   extra_special_tokens type : {type(val).__name__}")
    print(f"   extra_special_tokens value: {val}")

    if isinstance(val, list):
        cfg["extra_special_tokens"] = {}
        print("   ✅ Converted list → empty dict {}")
    elif isinstance(val, dict):
        print("   ✅ Already a dict — no change needed")
    else:
        cfg["extra_special_tokens"] = {}
        print(f"   ⚠️  Unexpected type {type(val).__name__} — reset to {{}}")

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Fixed tokenizer_config.json saved to:\n   {CONFIG_FILE}")

if __name__ == "__main__":
    main()

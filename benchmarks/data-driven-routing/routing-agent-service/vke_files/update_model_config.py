import argparse
import json
import os
import shutil
import sys
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update fields in scalable RL agent model_config.json"
    )
    parser.add_argument(
        "--final_model_dir",
        required=True,
        help="Path to the final_model directory containing model_config.json",
    )
    parser.add_argument(
        "--num_requests_per_episode",
        type=int,
        help="Override num_requests_per_episode",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Override num_iterations",
    )
    parser.add_argument(
        "--num_episodes_per_iteration",
        type=int,
        help="Override num_episodes_per_iteration",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        help="Override n_eval_episodes",
    )
    parser.add_argument(
        "--training_epochs",
        type=int,
        help="Override training_epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch_size",
    )
    return parser.parse_args()


def ensure_file(path: str) -> None:
    if not os.path.isfile(path):
        print(f"Error: file does not exist: {path}", file=sys.stderr)
        sys.exit(1)


def make_backup(json_path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{json_path}.bak.{ts}"
    shutil.copy2(json_path, backup_path)
    return backup_path


def load_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: str, data: dict) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    updates = {}

    def maybe_set(key: str, value):
        if value is None:
            return
        old = cfg.get(key)
        if old != value:
            cfg[key] = value
            updates[key] = {"old": old, "new": value}

    maybe_set("num_requests_per_episode", args.num_requests_per_episode)
    maybe_set("num_iterations", args.num_iterations)
    maybe_set("num_episodes_per_iteration", args.num_episodes_per_iteration)
    maybe_set("n_eval_episodes", args.n_eval_episodes)
    maybe_set("training_epochs", args.training_epochs)
    maybe_set("batch_size", args.batch_size)

    return updates


def main() -> None:
    args = parse_args()
    json_path = os.path.join(args.final_model_dir, "model_config.json")
    ensure_file(json_path)

    cfg = load_json(json_path)
    updates = apply_overrides(cfg, args)

    if not updates:
        print("No changes requested. Nothing to update.")
        return

    backup_path = make_backup(json_path)
    save_json(json_path, cfg)

    print("Updated model_config.json")
    print(f"- file: {json_path}")
    print(f"- backup: {backup_path}")
    print("- changes:")
    for key, change in updates.items():
        print(f"  {key}: {change['old']} -> {change['new']}")


if __name__ == "__main__":
    main()







import argparse
from pathlib import Path

from bicycle_sim.models.bicycle import generate


def main():
    parser = argparse.ArgumentParser(description="Generate bicycle prototype scenes")
    parser.add_argument("--config", type=Path, default=Path("configs/bicycle_v0.json"))
    parser.add_argument("--output", type=Path, default=Path("assets/bicycle"))
    args = parser.parse_args()
    for path in generate(args.config, args.output):
        print(path)


if __name__ == "__main__":
    main()

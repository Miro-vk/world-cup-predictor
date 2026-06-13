import argparse
import time
from pathlib import Path

from world_cup_predictor.simulate import (
    BRACKET_PATH,
    GROUPS_PATH,
    PROCESSED_PATH,
    monte_carlo,
    simulate_tournament,
)
from world_cup_predictor.train import MODEL_PATH

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "output"


def cmd_simulate(args: argparse.Namespace) -> None:
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "monte_carlo_results.csv"

    print(f"Running {args.n:,} simulations (seed={args.seed}) ...")
    t0 = time.time()
    df = monte_carlo(
        n=args.n,
        seed=args.seed,
        output_path=output_path,
    )
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s  ({elapsed / args.n * 1000:.1f} ms/sim)")
    print(f"Results saved to {output_path}\n")
    print(df.head(args.top).to_string(index=False))


def cmd_once(args: argparse.Namespace) -> None:
    print("Simulating one tournament ...")
    result = simulate_tournament()
    print(f"Champion:    {result['champion']}")
    print(f"Runner-up:   {result['runner_up']}")
    print(f"3rd place:   {result['third_place']}")
    print(f"4th place:   {result['fourth_place']}")
    print()
    print("Group winners:")
    for group, standings in sorted(result["group_standings"].items()):
        print(f"  Group {group}: {standings[0]['team']}")


def cmd_app(_args: argparse.Namespace) -> None:
    import subprocess, sys
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wc2026",
        description="2026 FIFA World Cup predictor",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # wc2026 simulate
    sim = sub.add_parser("simulate", help="Run Monte Carlo simulations")
    sim.add_argument("-n", type=int, default=5_000, help="Number of simulations (default: 5000)")
    sim.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    sim.add_argument("--output", type=str, default=None, help="Output CSV path (default: data/output/monte_carlo_results.csv)")
    sim.add_argument("--top", type=int, default=20, help="Number of teams to print (default: 20)")
    sim.set_defaults(func=cmd_simulate)

    # wc2026 once
    once = sub.add_parser("once", help="Run a single tournament and print the bracket result")
    once.set_defaults(func=cmd_once)

    # wc2026 app
    app = sub.add_parser("app", help="Launch the Streamlit web app")
    app.set_defaults(func=cmd_app)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
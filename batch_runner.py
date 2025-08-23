#!/usr/bin/env python3
import argparse, csv, os, time, importlib

def main():
    p = argparse.ArgumentParser(description="Batch-run Pacman_AI_LT2 episodes with telemetry.")
    p.add_argument("-n", "--num_games", type=int, default=10)
    p.add_argument("--seed0", type=int, default=12345)
    p.add_argument("--max_time", type=float, default=60.0, help="DNF cap per game (seconds)")
    p.add_argument("--max_moves", type=int, default=None, help="DNF cap per game (moves)")
    p.add_argument("--headless", action="store_true", help="Run without a window for speed")
    p.add_argument("--csv", default="pacman_runs.csv", help="Output CSV path")
    p.add_argument("--module", default="Pacman_AI_LT2", help="Which Pacman module to use")
    args = p.parse_args()

    # Dynamically import the chosen module
    try:
        pacman_mod = importlib.import_module(args.module)
    except ImportError as e:
        raise SystemExit(f"Error: Could not import module {args.module}: {e}")

    run_single_game_telemetry = pacman_mod.run_single_game_telemetry

    new_file = not os.path.exists(args.csv)
    fields = ["i","seed","status","score","elapsed_sec","moves",
              "pellets_total","pellets_eaten","completion_pct","ts"]
    wins = losses = dnfs = 0

    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()

        for i in range(args.num_games):
            seed = args.seed0 + i
            res = run_single_game_telemetry(
                layout_name="inline",
                seed=seed,
                max_time_sec=args.max_time,
                max_moves=args.max_moves,
                headless=args.headless
            )

            status = res["status"]
            if status == "WIN": wins += 1
            elif status == "LOSS": losses += 1
            else: dnfs += 1

            w.writerow({
                "i": i,
                "seed": res["seed"],
                "status": status,
                "score": res["score"],
                "elapsed_sec": res["elapsed_sec"],
                "moves": res["moves"],
                "pellets_total": res["pellets_total"],
                "pellets_eaten": res["pellets_eaten"],
                "completion_pct": res["completion_pct"],
                "ts": int(time.time()),
            })

    total = args.num_games or 1
    print(f"Games: {total} | Wins: {wins} | Losses: {losses} | DNF: {dnfs} | Win rate: {wins/total*100:.2f}%")

if __name__ == "__main__":
    main()

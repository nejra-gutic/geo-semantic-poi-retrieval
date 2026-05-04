import argparse
from src.utils.io import load_csv, save_csv
from src.preprocessing import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = load_csv(args.input)
    df_clean = pipeline.run(df)
    save_csv(df_clean, args.output)


if __name__ == "__main__":
    main()
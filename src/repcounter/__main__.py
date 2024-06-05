import argparse

from repcounter.movinet import MoviNet


def parse_args():
    parser = argparse.ArgumentParser(description="RepCounter")
    parser.add_argument(
        "-i", "--input", type=str, help="Path to the input video file", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output json file",
        default="output.json",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--start-frame", type=int, help="Start frame", default=0)
    parser.add_argument("--end-frame", type=int, help="End frame", default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    movinet = MoviNet()

    for pred in movinet(
        file_path=args.input, start_frame=args.start_frame, end_frame=args.end_frame
    ):
        print(pred)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
from pathlib import Path

from barcode_restoration.registry import available_methods, create_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified inference interface for barcode restoration methods."
    )

    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List currently integrated restoration methods.",
    )

    parser.add_argument(
        "--method",
        type=str,
        help="Restoration method to use.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional model/task variant.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the degraded input image.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the restored output image.",
    )

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device (default: cuda).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_methods:
        methods = available_methods()

        if not methods:
            print("No restoration methods are integrated yet.")
        else:
            for method in methods:
                print(method)

        return

    if not args.method:
        raise SystemExit("Error: --method is required.")

    if not args.input:
        raise SystemExit("Error: --input is required.")

    if not args.input.is_file():
        raise SystemExit(f"Error: input image does not exist: {args.input}")

    output_path = args.output

    if output_path is None:
        output_path = Path("outputs") / (
            f"{args.input.stem}_{args.method}{args.input.suffix}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    method = create_method(args.method)

    result = method.restore(
        input_path=args.input,
        output_path=output_path,
        variant=args.variant,
        device=args.device,
    )

    print(f"Method:  {result.method}")
    print(f"Input:   {result.input_path}")
    print(f"Output:  {result.output_path}")
    print(f"Time:    {result.elapsed_seconds:.3f} s")


if __name__ == "__main__":
    main()

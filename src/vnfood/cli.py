"""Command line: ``vnfood {report,predict,serve}``."""

from __future__ import annotations

import argparse
import os


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vnfood", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="figures + reports/results.md from Kaggle outputs")
    rp.add_argument("--kaggle-dir", default="kaggle/output/main")
    rp.add_argument("--report-dir", default="reports")
    pr = sub.add_parser("predict", help="top-5 dishes for image files")
    pr.add_argument("images", nargs="+")
    pr.add_argument("--model", default="artifacts/efficientnet_b0.onnx")
    pr.add_argument("--meta", default="artifacts/classes.json")
    sv = sub.add_parser("serve", help="upload page + REST API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    if args.cmd == "report":
        from vnfood.report import build_report

        print(f"report written to {build_report(args.kaggle_dir, args.report_dir)}")
    elif args.cmd == "predict":
        from vnfood.predict import FoodClassifier

        clf = FoodClassifier.load(args.model, args.meta)
        for path in args.images:
            with open(path, "rb") as fh:
                preds = clf.predict(fh.read())
            print(path, " | ".join(f"{q.name} {100 * q.probability:.1f}%" for q in preds))
    elif args.cmd == "serve":
        import uvicorn

        os.environ.setdefault("VNFOOD_MODEL", "artifacts/efficientnet_b0.onnx")
        uvicorn.run("vnfood.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

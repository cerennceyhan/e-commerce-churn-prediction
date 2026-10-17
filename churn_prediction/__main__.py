"""Pipeline entry point: ``python -m churn_prediction <step>``."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="churn_prediction", description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("scrape", help="scrape Trendyol reviews for the links in config/product_links.txt")
    sub.add_parser("features", help="step 1: compute base product metrics")
    extract = sub.add_parser("extract", help="step 2: extract LLM features with Claude")
    extract.add_argument("--max-products", type=int, default=None, help="limit products (for quick tests)")
    extract.add_argument("--delay", type=float, default=1.0, help="seconds between API calls")
    sub.add_parser("train", help="step 3: train XGBoost and write reports to outputs/")
    args = parser.parse_args(argv)

    # Imported lazily so e.g. `features` does not need selenium or xgboost installed.
    if args.step == "scrape":
        from churn_prediction import scraper
        scraper.main()
    elif args.step == "features":
        from churn_prediction import features
        features.main()
    elif args.step == "extract":
        from churn_prediction import llm_extraction
        llm_extraction.main(max_products=args.max_products, delay=args.delay)
    elif args.step == "train":
        from churn_prediction import train
        train.main()


if __name__ == "__main__":
    main()

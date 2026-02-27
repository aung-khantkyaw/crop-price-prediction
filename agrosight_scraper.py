import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

OUTPUT_COLUMNS = [
    "စဥ်",
    "အမျိုးအမည်",
    "ကုန်စည်ဒိုင်",
    "နေ့စွဲ",
    "အချင်အတွယ်",
    "စျေးနှုန်း (မြန်မာကျပ်)",
    "အတက်/အကျ",
    "%",
]


def set_page_in_url(base_url: str, page: int) -> str:
    parsed = urlparse(base_url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["page"] = str(page)
    new_query = urlencode(params, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def safe_filename(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^\w\-.]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "output"


def infer_output_prefix(url: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    shop = params.get("shop_id", "shop")
    crop = params.get("crop", "crop")
    date_from = params.get("from", "from")
    date_to = params.get("to", "to")
    return safe_filename(f"agrosight_shop{shop}_{crop}_{date_from}_to_{date_to}")


def extract_rows(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one("div.table-responsive.mt-4")
    if not container:
        return []

    table = container.find("table")
    if not table:
        return []

    tbody = table.find("tbody")
    trs = tbody.find_all("tr") if tbody else table.find_all("tr")

    rows = []
    for tr in trs:
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) < 8:
            continue

        row = {
            "စဥ်": tds[0],
            "အမျိုးအမည်": tds[1],
            "ကုန်စည်ဒိုင်": tds[2],
            "နေ့စွဲ": tds[3],
            "အချင်အတွယ်": tds[4],
            "စျေးနှုန်း (မြန်မာကျပ်)": tds[5],
            "အတက်/အကျ": tds[6],
            "%": tds[7],
        }
        rows.append(row)

    return rows


def scrape(url: str, max_page: int, timeout: int = 30) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    all_rows = []
    for page in range(1, max_page + 1):
        page_url = set_page_in_url(url, page)
        response = session.get(page_url, timeout=timeout)
        response.raise_for_status()
        all_rows.extend(extract_rows(response.text))

    return all_rows


def write_outputs(rows: list[dict], output_prefix: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    json_path = output_dir / f"{output_prefix}.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Agrosight price table into CSV and JSON."
    )
    parser.add_argument("--url", required=True, help="Agrosight search URL (can include page=1).")
    parser.add_argument("--max-page", type=int, required=True, help="Maximum page number to scrape (e.g., 17).")
    parser.add_argument("--output-prefix", help="Output filename prefix without extension.")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_page < 1:
        raise SystemExit("--max-page must be >= 1")

    output_prefix = args.output_prefix or infer_output_prefix(args.url)
    rows = scrape(args.url, args.max_page, timeout=args.timeout)
    csv_path, json_path = write_outputs(rows, output_prefix, Path(args.output_dir))

    print(f"Rows scraped: {len(rows)}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()

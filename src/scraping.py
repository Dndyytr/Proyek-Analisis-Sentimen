#!/usr/bin/env python3
"""Collect balanced English CS2 Steam reviews using Steam's GetAppReviews API.

Default output: data/raw/steam_reviews_main_10000_raw.csv
Default target: 5,000 positive + 5,000 negative reviews.

The script writes one CSV page at a time and checkpoints the next cursor, so it
can resume after interruption. It never overwrites the pilot dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

APP_ID = 730
API_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
SOURCE_API = "IUserReviewsService/GetAppReviews/v1"
FILTER_RECENT = 1
REVIEW_TYPE_POSITIVE = 1
REVIEW_TYPE_NEGATIVE = 2
PURCHASE_TYPE_ALL = 1
NUM_PER_PAGE = 100
MAX_RETRIES = 8
REQUEST_TIMEOUT = 45

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "steam_reviews_main_10000_raw.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "raw" / "checkpoint_main_10000.json"
DEFAULT_LOG = PROJECT_ROOT / "logs" / "scraping_main_10000.log"

FIELDNAMES = [
    "review_id",
    "app_id",
    "source_api",
    "retrieved_at_utc",
    "requested_review_type",
    "requested_filter",
    "requested_language",
    "requested_purchase_type",
    "steam_recommendation_label",
    "label_source",
    "sentiment",
    "review",
    "language",
    "voted_up",
    "timestamp_created",
    "timestamp_updated",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "author_steamid",
    "author_num_reviews",
    "author_playtime_forever",
    "author_playtime_last_two_weeks",
    "author_playtime_at_review",
    "query_total_matching",
]

CLASS_CONFIG = {
    REVIEW_TYPE_POSITIVE: {"label": "positif", "expected_voted_up": True},
    REVIEW_TYPE_NEGATIVE: {"label": "negatif", "expected_voted_up": False},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("steam_cs2_scraper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def build_initial_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": config,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "classes": {
            str(REVIEW_TYPE_POSITIVE): {
                "label": "positif",
                "cursor": "*",
                "pages": 0,
                "collected": 0,
                "total_matching": None,
                "status": "pending",
            },
            str(REVIEW_TYPE_NEGATIVE): {
                "label": "negatif",
                "cursor": "*",
                "pages": 0,
                "collected": 0,
                "total_matching": None,
                "status": "pending",
            },
        },
    }


def load_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        state = build_initial_checkpoint(config)
        save_checkpoint(checkpoint_path, state)
        return state

    with checkpoint_path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    if state.get("config") != config:
        raise RuntimeError(
            "Konfigurasi checkpoint berbeda dari konfigurasi sekarang. "
            "Jangan gunakan checkpoint yang sama untuk target/filter berbeda. "
            "Gunakan nama checkpoint baru atau pulihkan konfigurasi sebelumnya."
        )

    logger.info("Checkpoint ditemukan; proses akan dilanjutkan dari cursor tersimpan.")
    return state


def save_checkpoint(checkpoint_path: Path, state: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = utc_now()
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(checkpoint_path)


def load_existing_dataset(
    output_path: Path,
    logger: logging.Logger,
) -> tuple[set[str], dict[str, int]]:
    known_ids: set[str] = set()
    counts = {"positif": 0, "negatif": 0}

    if not output_path.exists() or output_path.stat().st_size == 0:
        return known_ids, counts

    with output_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"review_id", "steam_recommendation_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                f"Kolom file output tidak sesuai. Diperlukan: {sorted(required)}"
            )

        duplicate_rows = 0
        for row in reader:
            review_id = str(row.get("review_id", "")).strip()
            if not review_id:
                continue
            if review_id in known_ids:
                duplicate_rows += 1
                continue

            known_ids.add(review_id)
            label = str(row.get("steam_recommendation_label", "")).strip()
            if label in counts:
                counts[label] += 1

    if duplicate_rows:
        logger.warning(
            "File output sudah berisi %s baris dengan review_id duplikat; "
            "baris lama tidak dihapus otomatis.",
            duplicate_rows,
        )

    return known_ids, counts


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_has_data = output_path.exists() and output_path.stat().st_size > 0
    encoding = "utf-8" if already_has_data else "utf-8-sig"

    with output_path.open("a", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )
        if not already_has_data:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()


def fetch_page(
    session: requests.Session,
    request_payload: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    input_json = json.dumps(
        request_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                API_URL,
                params={"input_json": input_json},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Permintaan Steam gagal setelah {MAX_RETRIES} percobaan: {exc}"
                ) from exc
            wait_seconds = min(2 ** (attempt - 1), 90)
            logger.warning(
                "Gangguan koneksi (%s). Percobaan %s/%s; tunggu %s detik.",
                exc,
                attempt,
                MAX_RETRIES,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        retryable = response.status_code == 429 or response.status_code in {
            408, 500, 502, 503, 504
        }

        if retryable:
            if attempt == MAX_RETRIES:
                response.raise_for_status()
                raise RuntimeError("Steam terus mengembalikan respons yang dapat dicoba ulang.")

            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else 2 ** (attempt - 1)
            except ValueError:
                wait_seconds = 2 ** (attempt - 1)

            wait_seconds = min(max(wait_seconds, 1), 120)
            logger.warning(
                "HTTP %s dari Steam. Percobaan %s/%s; tunggu %.1f detik.",
                response.status_code,
                attempt,
                MAX_RETRIES,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Respons Steam bukan JSON yang valid: {response.text[:300]}"
            ) from exc

        service_response = body.get("response")
        if not isinstance(service_response, dict):
            raise RuntimeError(
                f"Struktur respons Steam tidak sesuai: {str(body)[:500]}"
            )

        return service_response

    raise RuntimeError("Permintaan Steam gagal tanpa respons yang dapat diproses.")


def make_output_row(
    item: dict[str, Any],
    review_type: int,
    total_matching: Any,
) -> dict[str, Any]:
    author = item.get("author") or {}
    if not isinstance(author, dict):
        author = {}

    voted_up = to_bool(item.get("voted_up"))
    label = "positif" if voted_up else "negatif"

    author_steamid = author.get("steamid")
    if author_steamid is not None:
        author_steamid = str(author_steamid)

    return {
        "review_id": str(item.get("recommendationid", "")),
        "app_id": APP_ID,
        "source_api": SOURCE_API,
        "retrieved_at_utc": utc_now(),
        "requested_review_type": review_type,
        "requested_filter": "recent",
        "requested_language": "english",
        "requested_purchase_type": "all",
        "steam_recommendation_label": label,
        "label_source": "steam_voted_up",
        # Kolom sentiment dipertahankan sebagai alias agar kompatibel
        # dengan sebagian analisis pilot; sumbernya tetap voted_up Steam.
        "sentiment": label,
        "review": item.get("review", ""),
        "language": item.get("language", "english"),
        "voted_up": voted_up,
        "timestamp_created": item.get("timestamp_created"),
        "timestamp_updated": item.get("timestamp_updated"),
        "votes_up": item.get("votes_up"),
        "votes_funny": item.get("votes_funny"),
        "weighted_vote_score": item.get("weighted_vote_score"),
        "steam_purchase": to_bool(item.get("steam_purchase")),
        "received_for_free": to_bool(item.get("received_for_free")),
        "written_during_early_access": to_bool(
            item.get("written_during_early_access")
        ),
        "author_steamid": author_steamid,
        "author_num_reviews": author.get("num_reviews"),
        "author_playtime_forever": author.get("playtime_forever"),
        "author_playtime_last_two_weeks": author.get("playtime_last_two_weeks"),
        "author_playtime_at_review": author.get("playtime_at_review"),
        "query_total_matching": total_matching,
    }


def collect_review_class(
    review_type: int,
    target_per_class: int,
    output_path: Path,
    checkpoint_path: Path,
    state: dict[str, Any],
    known_ids: set[str],
    counts: dict[str, int],
    session: requests.Session,
    logger: logging.Logger,
    delay_seconds: float,
) -> None:
    class_info = CLASS_CONFIG[review_type]
    label = class_info["label"]
    expected_voted_up = class_info["expected_voted_up"]
    class_state = state["classes"][str(review_type)]

    if counts[label] >= target_per_class:
        class_state["status"] = "complete"
        class_state["collected"] = counts[label]
        save_checkpoint(checkpoint_path, state)
        logger.info("Target %s sudah terpenuhi: %s/%s.", label, counts[label], target_per_class)
        return

    if class_state.get("status") == "exhausted":
        logger.warning(
            "Cursor kelas %s sebelumnya sudah mencapai halaman kosong. "
            "Data yang terkumpul: %s/%s.",
            label,
            counts[label],
            target_per_class,
        )
        return

    while counts[label] < target_per_class:
        cursor = class_state.get("cursor") or "*"
        request_payload = {
            "appid": APP_ID,
            "filter": FILTER_RECENT,
            "languages": ["english"],
            "review_type": review_type,
            "purchase_type": PURCHASE_TYPE_ALL,
            "num_per_page": NUM_PER_PAGE,
            "cursor": cursor,
        }

        logger.info(
            "Meminta kelas %s | terkumpul %s/%s | halaman %s | cursor aktif.",
            label,
            counts[label],
            target_per_class,
            class_state.get("pages", 0) + 1,
        )

        page = fetch_page(session, request_payload, logger)
        reviews = page.get("reviews") or []
        class_state["pages"] = int(class_state.get("pages", 0)) + 1

        if page.get("total_matching") is not None:
            class_state["total_matching"] = page.get("total_matching")

        if not reviews:
            class_state["status"] = "exhausted"
            class_state["collected"] = counts[label]
            save_checkpoint(checkpoint_path, state)
            logger.warning(
                "Steam mengembalikan halaman kosong untuk kelas %s. Terkumpul %s/%s.",
                label,
                counts[label],
                target_per_class,
            )
            break

        remaining = target_per_class - counts[label]
        page_rows: list[dict[str, Any]] = []
        duplicate_count = 0
        label_mismatch_count = 0
        missing_id_count = 0

        for item in reviews:
            review_id = str(item.get("recommendationid", "")).strip()
            if not review_id:
                missing_id_count += 1
                continue

            if review_id in known_ids:
                duplicate_count += 1
                continue

            voted_up = to_bool(item.get("voted_up"))
            if voted_up is None or voted_up != expected_voted_up:
                label_mismatch_count += 1
                continue

            row = make_output_row(item, review_type, class_state.get("total_matching"))
            page_rows.append(row)
            known_ids.add(review_id)

            if len(page_rows) >= remaining:
                break

        append_rows(output_path, page_rows)
        counts[label] += len(page_rows)
        class_state["collected"] = counts[label]

        next_cursor = page.get("cursor")

        if counts[label] >= target_per_class:
            class_state["status"] = "complete"
            if next_cursor:
                class_state["cursor"] = next_cursor
            save_checkpoint(checkpoint_path, state)
            logger.info(
                "Target kelas %s tercapai: %s/%s.",
                label,
                counts[label],
                target_per_class,
            )
            break

        if not next_cursor:
            class_state["status"] = "error_missing_cursor"
            save_checkpoint(checkpoint_path, state)
            raise RuntimeError(
                f"Respons kelas {label} berisi ulasan tetapi tidak memiliki cursor berikutnya. "
                "Checkpoint sudah disimpan; periksa log dan respons API."
            )

        if str(next_cursor) == str(cursor):
            class_state["status"] = "error_repeated_cursor"
            save_checkpoint(checkpoint_path, state)
            raise RuntimeError(
                f"Steam mengembalikan cursor yang sama untuk kelas {label}. "
                "Proses dihentikan agar tidak berulang tanpa akhir."
            )

        class_state["cursor"] = next_cursor
        class_state["status"] = "running"
        save_checkpoint(checkpoint_path, state)

        logger.info(
            "Halaman tersimpan: %s baris baru | duplikat dilewati: %s | "
            "voted_up tidak sesuai filter: %s | ID kosong: %s | total %s: %s/%s.",
            len(page_rows),
            duplicate_count,
            label_mismatch_count,
            missing_id_count,
            label,
            counts[label],
            target_per_class,
        )

        if delay_seconds > 0:
            time.sleep(delay_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scraper ulasan Steam CS2 menggunakan GetAppReviews API."
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=5000,
        help="Target jumlah ulasan positif dan negatif secara terpisah (default: 5000).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Lokasi CSV hasil scraping.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Lokasi file checkpoint JSON.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG,
        help="Lokasi file log.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.5,
        help="Jeda antarhalaman untuk mengurangi risiko rate limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_per_class < 1:
        raise ValueError("--target-per-class harus minimal 1.")
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds tidak boleh negatif.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.log_file)

    config = {
        "app_id": APP_ID,
        "filter": FILTER_RECENT,
        "language": "english",
        "review_type_values": [REVIEW_TYPE_POSITIVE, REVIEW_TYPE_NEGATIVE],
        "purchase_type": PURCHASE_TYPE_ALL,
        "num_per_page": NUM_PER_PAGE,
        "target_per_class": args.target_per_class,
    }

    state = load_checkpoint(args.checkpoint, config, logger)
    known_ids, counts = load_existing_dataset(args.output_csv, logger)

    if args.output_csv.exists() and not args.checkpoint.exists():
        logger.warning(
            "CSV sudah ada tetapi checkpoint belum ada. Scraper akan mulai dari cursor '*'; "
            "review_id yang sudah tersimpan akan dilewati."
        )

    logger.info("=" * 68)
    logger.info("Mulai scraping Steam CS2 (App ID %s).", APP_ID)
    logger.info("Endpoint: %s", API_URL)
    logger.info("Output: %s", args.output_csv)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Target per kelas: %s", args.target_per_class)
    logger.info("Jumlah review_id unik yang sudah ada: %s", len(known_ids))
    logger.info("Jumlah awal: positif=%s, negatif=%s", counts["positif"], counts["negatif"])
    logger.info("=" * 68)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "CS2-Sentiment-Research-Scraper/1.0 (educational project)"}
    )

    try:
        for review_type in (REVIEW_TYPE_POSITIVE, REVIEW_TYPE_NEGATIVE):
            collect_review_class(
                review_type=review_type,
                target_per_class=args.target_per_class,
                output_path=args.output_csv,
                checkpoint_path=args.checkpoint,
                state=state,
                known_ids=known_ids,
                counts=counts,
                session=session,
                logger=logger,
                delay_seconds=args.delay_seconds,
            )
    finally:
        session.close()

    logger.info("=" * 68)
    logger.info("HASIL AKHIR SCRAPING")
    logger.info("Positif: %s/%s", counts["positif"], args.target_per_class)
    logger.info("Negatif: %s/%s", counts["negatif"], args.target_per_class)
    logger.info("Total review_id unik: %s", len(known_ids))
    logger.info("Status positif: %s", state["classes"]["1"].get("status"))
    logger.info("Status negatif: %s", state["classes"]["2"].get("status"))
    logger.info("CSV: %s", args.output_csv)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("=" * 68)

    if counts["positif"] < args.target_per_class or counts["negatif"] < args.target_per_class:
        logger.warning(
            "Target belum seluruhnya tercapai. Periksa status kelas dan log. "
            "Jangan menghapus CSV/checkpoint jika ingin melanjutkan proses."
        )


if __name__ == "__main__":
    main()

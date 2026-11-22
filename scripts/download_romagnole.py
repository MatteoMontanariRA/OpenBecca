"""Download all available public-domain Romagnole playing cards from Wikimedia Commons.

Saves resized PNGs into ui/assets/cards/{SUIT}_{RANK}.png matching OpenBecca's
exact Card enum names, downloads the authentic card back (BACK.png), and
generates matching vintage-styled fallback cards for any missing Wikimedia uploads
(currently Re di Denari and Re di Bastoni).
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from openbecca.cards import ALL_CARDS, Card, Suit

USER_AGENT = "OpenBecca/0.1 (https://github.com/; recreational Beccacino AI UI)"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ui" / "assets" / "cards"

RANK_POLISH_TO_IT: dict[str, str] = {
    "as": "Asso",
    "dwa": "Due",
    "dwie": "Due",
    "trzy": "Tre",
    "cztery": "Quattro",
    "pięć": "Cinque",
    "sześć": "Sei",
    "siedem": "Sette",
    "paź": "Fante",
    "walet": "Fante",
    "jeździec": "Cavallo",
    "rycerz": "Cavallo",
    "król": "Re",
}

SUIT_POLISH_TO_IT: dict[str, str] = {
    "monet": "Denari",
    "monety": "Denari",
    "denarów": "Denari",
    "puszek": "Coppe",
    "puszki": "Coppe",
    "kielichów": "Coppe",
    "kielichy": "Coppe",
    "mieczy": "Spade",
    "miecze": "Spade",
    "maczug": "Bastoni",
    "maczugi": "Bastoni",
}


def _http_get_bytes(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 1.5 * (attempt + 1)
                print(f"  HTTP {exc.code} — retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed to download {url}")


def _fetch_category_files() -> list[dict]:
    pages: list[dict] = []
    gcmcontinue: str | None = None

    while True:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": "Category:Romagna_deck",
            "gcmlimit": "100",
            "gcmnamespace": "6",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "240",
            "format": "json",
        }
        if gcmcontinue:
            params["gcmcontinue"] = gcmcontinue

        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
        data = json.loads(_http_get_bytes(url).decode("utf-8"))
        batch = data.get("query", {}).get("pages", {})
        pages.extend(batch.values())

        cont = data.get("continue", {}).get("gcmcontinue")
        if not cont:
            break
        gcmcontinue = cont

    return pages


def _parse_romagnole_title(title: str, lookup: dict[str, Card]) -> Card | None:
    clean = title.removeprefix("File:").strip()
    lower = clean.lower()
    if "wzoru rzymskiego" not in lower:
        return None

    prefix = lower.split("z wzoru rzymskiego", 1)[0].strip()
    parts = prefix.split()
    if len(parts) < 2:
        return None

    rank_it = RANK_POLISH_TO_IT.get(parts[0])
    suit_it = SUIT_POLISH_TO_IT.get(parts[1])
    if not rank_it or not suit_it:
        return None

    return lookup.get(f"{rank_it} di {suit_it}")


def _load_serif_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _generate_fallback_card(card: Card, dest: Path, size: tuple[int, int]) -> None:
    """Create a vintage Romagnole-styled fallback PNG matching the deck's dimensions."""
    w, h = size
    img = Image.new("RGB", (w, h), "#f4ecd8")
    draw = ImageDraw.Draw(img)

    # Outer & inner parchment borders
    draw.rectangle([2, 2, w - 3, h - 3], outline="#2c2117", width=3)
    draw.rectangle([10, 10, w - 11, h - 11], outline="#7a5c38", width=2)
    draw.rectangle([15, 15, w - 16, h - 16], outline="#b89867", width=1)

    font_corner = _load_serif_font(max(18, w // 9))
    font_title = _load_serif_font(max(18, w // 10))
    font_sub = _load_serif_font(max(14, w // 14))

    # Corner indices
    draw.text((22, 20), "RE", fill="#8b1e24", font=font_corner)
    bbox = draw.textbbox((0, 0), "RE", font=font_corner)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((w - 22 - tw, h - 22 - th), "RE", fill="#8b1e24", font=font_corner)

    # Royal gold crown in upper center
    cx = w // 2
    cy_crown = int(h * 0.22)
    cw = int(w * 0.44)
    ch = int(h * 0.10)
    crown_pts = [
        (cx - cw // 2, cy_crown + ch // 2),
        (cx - cw // 2 - 4, cy_crown - ch // 3),
        (cx - cw // 5, cy_crown + ch // 8),
        (cx, cy_crown - ch // 2 - 6),
        (cx + cw // 5, cy_crown + ch // 8),
        (cx + cw // 2 + 4, cy_crown - ch // 3),
        (cx + cw // 2, cy_crown + ch // 2),
    ]
    draw.polygon(crown_pts, fill="#d4a017", outline="#4a310a")
    draw.rectangle(
        [cx - cw // 2, cy_crown + ch // 2 - 6, cx + cw // 2, cy_crown + ch // 2 + 6],
        fill="#8b1e24",
        outline="#4a310a",
        width=2,
    )
    for gx in (cx - cw // 4, cx, cx + cw // 4):
        draw.ellipse([gx - 4, cy_crown + ch // 2 - 4, gx + 4, cy_crown + ch // 2 + 4], fill="#f7e28b")

    # Central suit emblem
    cy = int(h * 0.52)
    if card.suit == Suit.COINS:
        r = int(w * 0.27)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#e0ab2a", outline="#6b4408", width=4)
        r2 = int(r * 0.78)
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill="#f4cf5d", outline="#8b1e24", width=2)
        for deg in range(0, 360, 45):
            rad = math.radians(deg)
            x1 = cx + int(math.cos(rad) * (r2 * 0.28))
            y1 = cy + int(math.sin(rad) * (r2 * 0.28))
            x2 = cx + int(math.cos(rad) * (r2 * 0.90))
            y2 = cy + int(math.sin(rad) * (r2 * 0.90))
            draw.line([(x1, y1), (x2, y2)], fill="#8b1e24", width=3)
        r3 = int(r * 0.26)
        draw.ellipse([cx - r3, cy - r3, cx + r3, cy + r3], fill="#8b1e24", outline="#f7e28b", width=2)
    elif card.suit == Suit.CLUBS:
        # Ornate Romagnole vertical club (Bastone)
        top_y = cy - int(h * 0.17)
        bot_y = cy + int(h * 0.17)
        club_pts = [
            (cx - 10, bot_y),
            (cx - 15, cy + 10),
            (cx - 24, top_y + 18),
            (cx - 16, top_y),
            (cx + 16, top_y),
            (cx + 24, top_y + 18),
            (cx + 15, cy + 10),
            (cx + 10, bot_y),
        ]
        draw.polygon(club_pts, fill="#2e6f3e", outline="#1b3b22")
        for band_y in (top_y + 30, cy, bot_y - 26):
            draw.rectangle([cx - 20, band_y - 5, cx + 20, band_y + 5], fill="#d4a017", outline="#4a310a", width=2)
        draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill="#8b1e24", outline="#f7e28b", width=2)

    # Bottom banner label
    banner_y = int(h * 0.77)
    draw.rectangle([24, banner_y, w - 24, banner_y + 44], fill="#2c2117", outline="#d4a017", width=2)
    label = card.italian_name.upper()
    lb = draw.textbbox((0, 0), label, font=font_title)
    lw, lh = lb[2] - lb[0], lb[3] - lb[1]
    draw.text(((w - lw) // 2, banner_y + (44 - lh) // 2 - 2), label, fill="#f7e7b4", font=font_title)

    sub = "ROMAGNOLE"
    sb = draw.textbbox((0, 0), sub, font=font_sub)
    sw = sb[2] - sb[0]
    draw.text(((w - sw) // 2, banner_y + 50), sub, fill="#6b5335", font=font_sub)

    img.save(dest, format="PNG")


def download_romagnole_deck() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name_to_card = {c.italian_name: c for c in ALL_CARDS}

    print("Querying Wikimedia Commons (Category:Romagna_deck)...")
    pages = _fetch_category_files()

    matched: dict[Card, tuple[str, str]] = {}
    back_urls: tuple[str, str] | None = None

    for page in pages:
        title = page.get("title", "")
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        img_url = info.get("thumburl") or info.get("url")
        if not img_url:
            continue

        if "koszulka" in title.lower():
            back_urls = (img_url, info.get("url", img_url))
            continue

        card = _parse_romagnole_title(title, name_to_card)
        if card is not None:
            matched[card] = (img_url, info.get("url", img_url))

    # Download authentic Romagnole card back (BACK.png)
    back_dest = OUTPUT_DIR / "BACK.png"
    if back_urls and not (back_dest.exists() and back_dest.stat().st_size > 2000):
        try:
            raw = _http_get_bytes(back_urls[0])
        except Exception:
            raw = _http_get_bytes(back_urls[1])
        back_dest.write_bytes(raw)
        print(f"  [BACK] Authentic Romagnole card back -> {back_dest.name}")

    print(f"Matched {len(matched)}/40 Romagnole cards on Wikimedia Commons.")
    downloaded = 0
    for card in sorted(matched, key=lambda c: c.sort_key()):
        dest = OUTPUT_DIR / f"{card.suit.name}_{card.rank.name}.png"
        if dest.exists() and dest.stat().st_size > 2000:
            downloaded += 1
            continue

        thumb_url, full_url = matched[card]
        try:
            raw = _http_get_bytes(thumb_url)
        except Exception:
            raw = _http_get_bytes(full_url)

        dest.write_bytes(raw)
        downloaded += 1
        print(f"  [{downloaded:2d}/40] {card.italian_name} -> {dest.name}")
        time.sleep(0.15)

    # Match exact dimensions of downloaded deck for any missing Wikimedia cards
    ref_size = (240, 448)
    ref_path = OUTPUT_DIR / "CUPS_KING.png"
    if ref_path.exists():
        with Image.open(ref_path) as ref_img:
            ref_size = ref_img.size

    missing = [c for c in ALL_CARDS if c not in matched]
    for card in missing:
        dest = OUTPUT_DIR / f"{card.suit.name}_{card.rank.name}.png"
        if not (dest.exists() and dest.stat().st_size > 2000):
            _generate_fallback_card(card, dest, ref_size)
            print(f"  [GEN ] {card.italian_name} (missing on Commons) -> {dest.name}")

    total_cards = sum(
        1 for c in ALL_CARDS if (OUTPUT_DIR / f"{c.suit.name}_{c.rank.name}.png").exists()
    )
    print(
        f"\nDone! {total_cards}/40 cards + BACK.png ready in {OUTPUT_DIR}."
    )
    if missing:
        names = ", ".join(f"{c.suit.name}_{c.rank.name}.png" for c in missing)
        print(
            f"Note: {names} were generated as matching vintage fallbacks because they are "
            "missing from Wikimedia Commons. You can drop real scans with those filenames anytime."
        )


if __name__ == "__main__":
    download_romagnole_deck()
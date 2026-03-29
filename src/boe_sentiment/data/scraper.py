"""
Scraper for Bank of England MPC minutes.
Downloads PDFs from the BoE website and caches extracted text locally.
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pdfplumber

logger = logging.getLogger(__name__)

BOE_BASE = "https://www.bankofengland.co.uk"


@dataclass
class MPCDocument:
    date: datetime
    url: str
    text: str
    n_words: int


KNOWN_MINUTES = {
    "2024-11": "/monetary-policy-summary-and-minutes/2024/november-2024",
    "2024-09": "/monetary-policy-summary-and-minutes/2024/september-2024",
    "2024-08": "/monetary-policy-summary-and-minutes/2024/august-2024",
    "2024-06": "/monetary-policy-summary-and-minutes/2024/june-2024",
    "2024-05": "/monetary-policy-summary-and-minutes/2024/may-2024",
    "2024-03": "/monetary-policy-summary-and-minutes/2024/march-2024",
    "2024-02": "/monetary-policy-summary-and-minutes/2024/february-2024",
    "2023-12": "/monetary-policy-summary-and-minutes/2023/december-2023",
    "2023-11": "/monetary-policy-summary-and-minutes/2023/november-2023",
    "2023-09": "/monetary-policy-summary-and-minutes/2023/september-2023",
    "2023-08": "/monetary-policy-summary-and-minutes/2023/august-2023",
    "2023-06": "/monetary-policy-summary-and-minutes/2023/june-2023",
    "2023-05": "/monetary-policy-summary-and-minutes/2023/may-2023",
    "2023-03": "/monetary-policy-summary-and-minutes/2023/march-2023",
    "2023-02": "/monetary-policy-summary-and-minutes/2023/february-2023",
    "2022-12": "/monetary-policy-summary-and-minutes/2022/december-2022",
    "2022-11": "/monetary-policy-summary-and-minutes/2022/november-2022",
    "2022-09": "/monetary-policy-summary-and-minutes/2022/september-2022",
    "2022-08": "/monetary-policy-summary-and-minutes/2022/august-2022",
    "2022-06": "/monetary-policy-summary-and-minutes/2022/june-2022",
    "2022-05": "/monetary-policy-summary-and-minutes/2022/may-2022",
    "2022-03": "/monetary-policy-summary-and-minutes/2022/march-2022",
    "2022-02": "/monetary-policy-summary-and-minutes/2022/february-2022",
    "2021-12": "/monetary-policy-summary-and-minutes/2021/december-2021",
    "2021-11": "/monetary-policy-summary-and-minutes/2021/november-2021",
    "2021-09": "/monetary-policy-summary-and-minutes/2021/september-2021",
    "2021-08": "/monetary-policy-summary-and-minutes/2021/august-2021",
    "2021-06": "/monetary-policy-summary-and-minutes/2021/june-2021",
    "2021-05": "/monetary-policy-summary-and-minutes/2021/may-2021",
    "2021-03": "/monetary-policy-summary-and-minutes/2021/march-2021",
    "2021-02": "/monetary-policy-summary-and-minutes/2021/february-2021",
    "2020-12": "/monetary-policy-summary-and-minutes/2020/december-2020",
    "2020-11": "/monetary-policy-summary-and-minutes/2020/november-2020",
    "2020-09": "/monetary-policy-summary-and-minutes/2020/september-2020",
    "2020-08": "/monetary-policy-summary-and-minutes/2020/august-2020",
    "2020-06": "/monetary-policy-summary-and-minutes/2020/june-2020",
    "2020-05": "/monetary-policy-summary-and-minutes/2020/may-2020",
    "2020-03": "/monetary-policy-summary-and-minutes/2020/march-2020",
    "2020-01": "/monetary-policy-summary-and-minutes/2020/january-2020",
    "2019-12": "/monetary-policy-summary-and-minutes/2019/december-2019",
    "2019-11": "/monetary-policy-summary-and-minutes/2019/november-2019",
    "2019-09": "/monetary-policy-summary-and-minutes/2019/september-2019",
    "2019-08": "/monetary-policy-summary-and-minutes/2019/august-2019",
    "2019-06": "/monetary-policy-summary-and-minutes/2019/june-2019",
    "2019-05": "/monetary-policy-summary-and-minutes/2019/may-2019",
    "2019-03": "/monetary-policy-summary-and-minutes/2019/march-2019",
    "2018-12": "/monetary-policy-summary-and-minutes/2018/december-2018",
    "2018-11": "/monetary-policy-summary-and-minutes/2018/november-2018",
    "2018-09": "/monetary-policy-summary-and-minutes/2018/september-2018",
    "2018-08": "/monetary-policy-summary-and-minutes/2018/august-2018",
    "2018-06": "/monetary-policy-summary-and-minutes/2018/june-2018",
    "2018-05": "/monetary-policy-summary-and-minutes/2018/may-2018",
    "2018-03": "/monetary-policy-summary-and-minutes/2018/march-2018",
}


class MPCScraper:
    """Fetches and caches MPC minutes from the Bank of England website."""

    def __init__(self, cache_dir: str = "data/raw", delay: float = 1.5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BoE-Sentiment-Research/0.1"})

    def fetch_minutes(self, start_year: int = 2018, end_year: int = 2024) -> list:
        docs = []
        keys = sorted(k for k in KNOWN_MINUTES if start_year <= int(k[:4]) <= end_year)
        for key in keys:
            try:
                doc = self._fetch_one(key)
                if doc:
                    docs.append(doc)
                    logger.info(f"Fetched {key}: {doc.n_words} words")
                time.sleep(self.delay)
            except Exception as e:
                logger.warning(f"Failed to fetch {key}: {e}")
        logger.info(f"Fetched {len(docs)} documents")
        return docs

    def _fetch_one(self, month_key: str):
        cache_path = self.cache_dir / f"mpc_{month_key}.txt"
        pdf_cache = self.cache_dir / f"mpc_{month_key}.pdf"
        date = datetime.strptime(month_key, "%Y-%m")

        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
            return MPCDocument(
                date=date,
                url=KNOWN_MINUTES[month_key],
                text=text,
                n_words=len(text.split()),
            )

        url = BOE_BASE + KNOWN_MINUTES[month_key]
        page = self.session.get(url, timeout=30)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")

        pdf_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "minutes" in href.lower() and href.endswith(".pdf"):
                pdf_link = href if href.startswith("http") else BOE_BASE + href
                break

        if pdf_link:
            text = self._extract_pdf_text(pdf_link, pdf_cache)
        else:
            text = self._extract_html_text(soup)

        if not text or len(text.split()) < 100:
            logger.warning(f"Insufficient text for {month_key}")
            return None

        cache_path.write_text(text, encoding="utf-8")
        return MPCDocument(date=date, url=url, text=text, n_words=len(text.split()))

    def _extract_pdf_text(self, pdf_url: str, cache_path: Path) -> str:
        if not cache_path.exists():
            resp = self.session.get(pdf_url, timeout=60)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        parts = []
        with pdfplumber.open(cache_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)

    def _extract_html_text(self, soup: BeautifulSoup) -> str:
        content = (
            soup.find("div", class_="page-content")
            or soup.find("article")
            or soup.find("main")
        )
        if content:
            for tag in content.find_all(["nav", "header", "footer", "script", "style"]):
                tag.decompose()
            return content.get_text(separator=" ", strip=True)
        return soup.get_text(separator=" ", strip=True)

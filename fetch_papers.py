import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time


def fetch_arxiv_papers(topic: str, max_results: int = 6) -> list[dict]:
    """Fetch papers from arXiv based on a research topic."""
    
    base_url = "http://export.arxiv.org/api/query?"
    query = urllib.parse.quote(topic)
    params = f"search_query=all:{query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    url = base_url + params

    # Try up to 3 times with increasing timeout
    last_error = None
    for attempt, timeout in enumerate([20, 30, 45], 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read().decode("utf-8")
            break  # success
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(2)  # wait before retry
    else:
        raise Exception(f"Failed to fetch from arXiv after 3 attempts: {last_error}")

    # Parse XML
    root = ET.fromstring(data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", namespace):
        title_el = entry.find("atom:title", namespace)
        summary_el = entry.find("atom:summary", namespace)
        published_el = entry.find("atom:published", namespace)

        authors = []
        for author in entry.findall("atom:author", namespace):
            name_el = author.find("atom:name", namespace)
            if name_el is not None:
                authors.append(name_el.text)

        link = ""
        for l in entry.findall("atom:link", namespace):
            if l.attrib.get("type") == "text/html":
                link = l.attrib.get("href", "")

        if title_el is not None and summary_el is not None:
            papers.append({
                "title": title_el.text.strip().replace("\n", " "),
                "summary": summary_el.text.strip().replace("\n", " "),
                "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                "published": published_el.text[:10] if published_el is not None else "N/A",
                "link": link
            })

    return papers

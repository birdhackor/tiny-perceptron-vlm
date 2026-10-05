"""Inspect current local W.1 and its 1.1 destinations using real Chromium."""
from pathlib import Path
import hashlib
import json
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8788/"
records = {}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1250, "height": 950}, device_scale_factor=1)
    page = context.new_page()
    build = context.request.get(BASE + "build-info.json")
    records["build_info"] = build.json()
    response = page.goto(BASE + "first-steps.html#W.1", wait_until="networkidle")
    records["W.1"] = {"url": page.url, "status": response.status, "title": page.title()}
    page.locator('[id="W.1"]').scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / "W.1-top.png"))
    owned_text = page.evaluate("""() => {
      const heading = document.getElementById('W.1');
      const nodes = []; let node = heading;
      while (node && (node === heading || node.tagName !== 'H2')) {
        nodes.push(node.innerText); node = node.nextElementSibling;
      }
      return nodes.join('\n');
    }""")
    records["W.1"]["owned_rendered_text"] = owned_text
    records["W.1"]["necessary_links"] = page.locator('a[href="1.1.html"]').evaluate_all("els => els.map(e=>({text:e.innerText,href:e.href}))")
    page.get_by_text('因此更改前面的輸入後', exact=False).first.scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / "W.1-kernel-exercise.png"))
    response = page.goto(BASE + "1.1.html", wait_until="networkidle")
    records["1.1"] = {"url": page.url, "status": response.status, "title": page.title(),
                       "body_text": page.locator('article').inner_text()}
    actions = page.locator('.actions a').evaluate_all("els => els.map(e=>({text:e.innerText,href:e.href,download:e.hasAttribute('download')}))")
    records["1.1"]["actions"] = actions
    assert len(actions) == 2
    page.locator('.actions').scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / "1.1-destinations.png"))
    downloaded = context.request.get(actions[1]["href"])
    raw = downloaded.body()
    expected = (ROOT / "notebooks/01/1.1.ipynb").read_bytes()
    records["download"] = {"url": actions[1]["href"], "status": downloaded.status,
                           "sha256": hashlib.sha256(raw).hexdigest(), "matches_current_notebook": raw == expected,
                           "metadata": json.loads(raw)["metadata"], "cells": len(json.loads(raw)["cells"])}
    assert downloaded.status == 200 and raw == expected
    response = page.goto(BASE + "figures/character_ids.svg", wait_until="networkidle")
    page.screenshot(path=str(OUT / "character_ids.render.png"), full_page=True)
    records["necessary_figure"] = {"url": page.url, "status": response.status,
                                   "sha256": hashlib.sha256(response.body()).hexdigest()}
    try:
        response = page.goto(actions[0]["href"], wait_until="domcontentloaded", timeout=25000)
        page.wait_for_timeout(1500)
        records["colab_destination"] = {"requested": actions[0]["href"], "url": page.url,
                                        "status": response.status if response else None,
                                        "title": page.title(), "visible_text": page.locator('body').inner_text()[:3500],
                                        "runtime_executed": False, "authenticated": False}
        page.screenshot(path=str(OUT / "colab-destination.png"))
    except Exception as e:
        records["colab_destination"] = {"requested": actions[0]["href"], "error": str(e),
                                        "runtime_executed": False, "authenticated": False}
    records["browser"] = {"chromium": browser.version, "playwright": "1.63.0", "mode": "headless", "os": sys.platform}
    browser.close()
(OUT / "own-browser-results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"build_info": records["build_info"], "download": records["download"],
                  "actions": records["1.1"]["actions"], "colab_destination": records["colab_destination"],
                  "browser": records["browser"]}, ensure_ascii=False, indent=2))

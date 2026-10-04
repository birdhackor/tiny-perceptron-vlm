from pathlib import Path
import json
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent
PREVIEW = 'http://127.0.0.1:8784/training.html#T.8'
receipt = {'engine': 'actual Chromium via installed Playwright', 'preview_requested': PREVIEW, 'actions': []}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1000})
    page = context.new_page()
    response = page.goto(PREVIEW, wait_until='domcontentloaded', timeout=45000)
    heading = page.locator('[id="T.8"]')
    heading.scroll_into_view_if_needed()
    receipt['preview_status'] = response.status
    receipt['preview_actual_url'] = page.url
    receipt['preview_heading'] = heading.inner_text()
    section = heading.evaluate(r'''node => {
        const parts = [], links = [];
        for (let n = node; n; n = n.nextElementSibling) {
            if (n !== node && n.tagName === 'H2') break;
            parts.push(n.innerText);
            for (const a of n.querySelectorAll('a')) links.push({text:a.innerText, href:a.href});
        }
        return {text:parts.join('\n\n'), links};
    }''')
    (ART / 'preview-T.8-rendered-text.txt').write_text(section['text'])
    (ART / 'preview-T.8-links.json').write_text(json.dumps(section['links'], ensure_ascii=False, indent=2) + '\n')
    page.screenshot(path=str(ART / 'preview-T.8-heading.png'))
    receipt['actions'].append({'action': 'goto actual pinned preview and scroll to T.8 heading', 'screenshot': str(ART / 'preview-T.8-heading.png')})
    target = page.get_by_role('link', name='2.14的max_memory_allocated文件', exact=True)
    target.scroll_into_view_if_needed()
    receipt['official_link_text'] = target.inner_text()
    receipt['official_link_href'] = target.get_attribute('href')
    page.screenshot(path=str(ART / 'preview-T.8-official-link.png'))
    receipt['actions'].append({'action': 'inspect visible official API link in assigned section', 'screenshot': str(ART / 'preview-T.8-official-link.png')})
    receipt['official_link_target'] = target.get_attribute('target')
    responses = []
    context.on('response', lambda resp: responses.append({'url': resp.url, 'status': resp.status}) if resp.request.resource_type == 'document' else None)
    if target.get_attribute('target') == '_blank':
        with page.expect_popup() as popup:
            target.click()
        page = popup.value
    else:
        target.click()
    page.wait_for_load_state('domcontentloaded', timeout=45000)
    receipt['official_actual_url'] = page.url
    receipt['official_document_responses'] = responses
    receipt['official_title'] = page.title()
    receipt['official_h1'] = page.locator('h1').all_inner_texts()
    main = page.locator('article').first
    if main.count() == 0:
        main = page.locator('main').first
    if main.count() == 0:
        main = page.locator('[role="main"]').first
    receipt['official_article_locator_available'] = main.count() > 0
    if main.count():
        text = main.inner_text()
        (ART / 'official-api-rendered-article.txt').write_text(text)
        receipt['official_article_saved'] = str(ART / 'official-api-rendered-article.txt')
    if page.locator('h1').count():
        page.locator('h1').first.scroll_into_view_if_needed()
    page.screenshot(path=str(ART / 'official-api-destination.png'))
    receipt['actions'].append({'action': 'actual click from T.8 to official API destination', 'screenshot': str(ART / 'official-api-destination.png')})
    receipt['browser_version'] = browser.version
    browser.close()
(ART / 'browser-navigation-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
if (ART / 'official-api-rendered-article.txt').exists():
    print((ART / 'official-api-rendered-article.txt').read_text())

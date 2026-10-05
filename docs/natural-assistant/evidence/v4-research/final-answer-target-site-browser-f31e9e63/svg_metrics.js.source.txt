async ({url, renderedWidth, renderedHeight}) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`SVG fetch failed: ${response.status}`);
  const source = await response.text();
  const parsed = new DOMParser().parseFromString(source, 'image/svg+xml');
  if (parsed.querySelector('parsererror')) throw new Error('Invalid SVG XML');
  const svg = document.importNode(parsed.documentElement, true);
  const values = svg.getAttribute('viewBox').trim().split(/[\s,]+/).map(Number);
  if (values.length !== 4 || values.some(x => !Number.isFinite(x)) || values[2] <= 0 || values[3] <= 0) throw new Error('Missing bounded SVG viewBox');
  const [left, top, width, height] = values;
  const probe = document.createElement('div');
  probe.style.cssText = 'position:absolute;left:-10000px;top:0;visibility:hidden;pointer-events:none';
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  probe.appendChild(svg);
  document.body.appendChild(probe);
  await document.fonts.ready;
  const scale = Math.min(renderedWidth / width, renderedHeight / height);
  const rootInverse = svg.getCTM().inverse();
  function nativeBounds(node) {
    const local = node.getBBox();
    const transform = rootInverse.multiply(node.getCTM());
    const corners = [
      [local.x, local.y],
      [local.x + local.width, local.y],
      [local.x, local.y + local.height],
      [local.x + local.width, local.y + local.height],
    ].map(([x, y]) => new DOMPoint(x, y).matrixTransform(transform));
    const xs = corners.map(p => p.x), ys = corners.map(p => p.y);
    return {x:Math.min(...xs), y:Math.min(...ys),
      width:Math.max(...xs)-Math.min(...xs), height:Math.max(...ys)-Math.min(...ys),
      font_scale:Math.hypot(transform.c, transform.d)};
  }
  const rows = Array.from(svg.querySelectorAll('text')).map(node => {
    const bounds = nativeBounds(node);
    const computed = getComputedStyle(node);
    const size = Number.parseFloat(computed.fontSize) * bounds.font_scale;
    const text = node.textContent;
    const captions = ['本圖不是完整圖文模型的序列長度。','這是資料卡示意，不是實際資料數量。'];
    const conclusions = ['第一個答案由前面的助理邊界預測','先切來源家族，再製作衍生題。','先核對聽寫，再核對兩份完整回答。'];
    const titles = ['問題留在輸入，答案才計分','同一來源的變體，一起切分','先聽寫，再回答：分開核對','同一張訓練照片，可問不同問題'];
    const role = captions.includes(text) ? 'caption' : conclusions.includes(text) ? 'main_conclusion' : titles.includes(text) ? 'main_title' : 'diagram_label_or_note';
    return {text, text_role:role, native_font_px:size, rendered_font_px:size*scale,
      declared_font_family:computed.fontFamily,
      bbox:{x:bounds.x,y:bounds.y,width:bounds.width,height:bounds.height},
      outside_viewbox:bounds.x < left-1 || bounds.y < top-1 || bounds.x+bounds.width > left+width+1 || bounds.y+bounds.height > top+height+1};
  });
  const images = Array.from(svg.querySelectorAll('image')).map(node => {
    const bounds = nativeBounds(node);
    return {bbox:{x:bounds.x,y:bounds.y,width:bounds.width,height:bounds.height},
      preserve_aspect_ratio:node.getAttribute('preserveAspectRatio'),
      embedded:(node.getAttribute('href') || node.getAttributeNS('http://www.w3.org/1999/xlink','href') || '').startsWith('data:'),
      outside_viewbox:bounds.x < left-1 || bounds.y < top-1 || bounds.x+bounds.width > left+width+1 || bounds.y+bounds.height > top+height+1};
  });
  probe.remove();
  return {viewbox:values, rendered_width:renderedWidth, rendered_height:renderedHeight, scale,
    min_native_font_px:rows.length ? Math.min(...rows.map(x=>x.native_font_px)) : null,
    min_rendered_font_px:rows.length ? Math.min(...rows.map(x=>x.rendered_font_px)) : null,
    text:rows, images, text_outside_viewbox:rows.filter(x=>x.outside_viewbox), image_outside_viewbox:images.filter(x=>x.outside_viewbox),
    scope:'SVG text/image bounding boxes transformed from node-local coordinates into the root viewBox, and computed font metrics including local transforms at actual rendered image scale; screenshots needed for visual legibility and font-face confirmation'};
}

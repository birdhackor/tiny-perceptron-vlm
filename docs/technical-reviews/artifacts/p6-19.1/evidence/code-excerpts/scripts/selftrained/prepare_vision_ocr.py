scripts/selftrained/prepare_vision_ocr.py:L30-L36
30: REPO = Path(__file__).resolve().parents[2]
31: FASHION_REVISION = "b2617bb6d3ffa2e429640350f613e3291e10b141"
32: NOTO_REVISION = "f8d157532fbfaeda587e826d4cd5b21a49186f7c"
33: VERSION = "phase5-vision-ocr-v1"
34: SEED = 20261006
35: CHARACTERS = "大小上下左右開關入出人口"
36: FASHION_CLASSES = {1: (0, "褲子"), 8: (1, "包"), 9: (2, "短靴")}

scripts/selftrained/prepare_vision_ocr.py:L167-L210
167: def prepare_vision(data: Path, counts: dict[str, int]) -> tuple[list[dict], dict]:
168:     official_train, train_labels = load_idx(data, "train")
169:     official_test, test_labels = load_idx(data, "t10k")
170:     selected = defaultdict(lambda: defaultdict(list))
171:     sources = []
172:     used_pixels: set[str] = set()
173:     skipped_duplicates = Counter()
174:     for label, (local_id, _) in FASHION_CLASSES.items():
175:         for original_split, pixels, labels, destinations in [
176:             ("train", official_train, train_labels, ["train", "validation"]),
177:             ("t10k", official_test, test_labels, ["test"]),
178:         ]:
179:             indices = np.flatnonzero(labels == label).tolist()
180:             random.Random(SEED + label + (100 if original_split == "t10k" else 0)).shuffle(indices)
181:             cursor = 0
182:             for split in destinations:
183:                 while len(selected[split][local_id]) < counts[split]:
184:                     if cursor == len(indices):
185:                         raise ValueError(f"Not enough unique original sources for {split}/{label}")
186:                     index = indices[cursor]
187:                     cursor += 1
188:                     image_hash = digest(pixels[index].tobytes())
189:                     if image_hash in used_pixels:
190:                         skipped_duplicates[f"{original_split}:{label}"] += 1
191:                         continue
192:                     used_pixels.add(image_hash)
193:                     source_id = f"fashion-mnist:{original_split}:{index:05d}"
194:                     selected[split][local_id].append((source_id, pixels[index]))
195:                     sources.append(
196:                         {
197:                             "source_id": source_id,
198:                             "official_split": original_split,
199:                             "original_index": index,
200:                             "split": split,
201:                             "original_label": label,
202:                             "local_label": local_id,
203:                             "pixel_sha256": image_hash,
204:                         }
205:                     )
206:     rows = []
207:     scenes = []
208:     for split in ["train", "validation", "test"]:
209:         for local_id, name in [value for value in FASHION_CLASSES.values()]:
210:             for source_id, pixels in selected[split][local_id]:

scripts/selftrained/prepare_vision_ocr.py:L329-L386
329: def prepare_ocr(data: Path, fonts: dict[str, Path], heldout_single_chars: bool) -> tuple[list[dict], dict]:
330:     rows, renders = [], []
331:     pools = word_pools()
332:     variants = {
333:         "train": [("Sans", 24, 255, 8, 4), ("Sans", 26, 255, 12, 8)],
334:         "validation": [("Sans", 22, 240, 18, 8), ("Serif", 24, 255, 18, 8)],
335:         "test": [("Sans", 28, 232, 30, 12), ("Serif", 26, 255, 30, 12)],
336:     }
337:     for split in ["train", "validation", "test"]:
338:         words = pools[split]
339:         if len(words) % 2:
340:             raise ValueError("OCR pool must have even size for same-image ROI pairs")
341:         for index in range(0, len(words), 2):
342:             texts = words[index : index + 2]
343:             chosen = [variants[split][(index // 2) % 2]] if split == "train" else variants[split]
344:             for family, size, background, x, y in chosen:
345:                 font = ImageFont.truetype(str(fonts[family]), size)
346:                 image = Image.new("L", (192, 96), background)
347:                 draw = ImageDraw.Draw(image)
348:                 glyphs = [
349:                     draw_line(draw, text, font, x, y + row * 42, (index % 3) - 1) for row, text in enumerate(texts)
350:                 ]
351:                 relative = save_image(data, "ocr", image)
352:                 source_id = opaque("ocr-render", split, texts, family, size, background, x, y, index)
353:                 group = opaque("ocr-composition", split, texts)
354:                 for row, text in enumerate(texts):
355:                     roi = [x, y + row * 42, x + len(text) * 36, y + row * 42 + 36]
356:                     rows.append(
357:                         record(
358:                             split,
359:                             "ocr",
360:                             group,
361:                             f"請讀出區域 {roi} 內的字。",
362:                             text,
363:                             relative,
364:                             roi=roi,
365:                             supervision={
366:                                 "ocr_text": text,
367:                                 "glyph_boxes": glyphs[row],
368:                                 "render_source_id": source_id,
369:                                 "case": "composition",
370:                                 "pair_id": source_id,
371:                                 "control_type": "roi",
372:                                 "font_family": family,
373:                                 "font_size": size,
374:                                 "background": background,
375:                                 "condition": "unseen_font" if family == "Serif" else "size_background_layout",
376:                             },
377:                         )
378:                     )
379:                 renders.append(
380:                     {
381:                         "render_source_id": source_id,
382:                         "group_id": group,
383:                         "split": split,
384:                         "image": relative,
385:                         "font_family": family,
386:                         "font_size": size,

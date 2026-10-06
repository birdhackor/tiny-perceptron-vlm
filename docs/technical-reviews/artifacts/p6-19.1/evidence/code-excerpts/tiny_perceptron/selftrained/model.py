tiny_perceptron/selftrained/model.py:L134-L151
134:     def __init__(self, config):
135:         language_config = ModelConfig(
136:             vocab_size=config.vocab_size,
137:             width=config.width,
138:             layers=config.layers,
139:             heads=config.heads,
140:             max_length=config.max_length,
141:             norm="rms",
142:             rotary=True,
143:             tied=config.tied,
144:             kv_heads=config.kv_heads,
145:             backend=config.backend,
146:             experts=0,
147:             top_k=config.top_k,
148:         )
149:         super().__init__(language_config)
150:         self.blocks = nn.ModuleList([MaskedBlock(language_config, config) for _ in range(config.layers)])
151:         self.config.experts = config.experts if config.architecture == "moe" else 0

tiny_perceptron/selftrained/model.py:L189-L282
189:     def __init__(self, width):
190:         super().__init__()
191:         self.cnn = nn.Sequential(
192:             nn.Conv2d(1, 16, 3, stride=2, padding=1),
193:             nn.GELU(),
194:             nn.Conv2d(16, 32, 3, stride=2, padding=1),
195:             nn.GELU(),
196:             nn.Conv2d(32, 64, 3, stride=2, padding=1),
197:             nn.GELU(),
198:             nn.AdaptiveAvgPool2d((2, 2)),
199:         )
200:         self.projection = nn.Linear(64 * 2 * 2, width)
201:         self.head = nn.Linear(width, 3)
202:         self.symbol_projection = nn.Linear(3, width, bias=False)
203:         self.coordinates = nn.Linear(2, width, bias=False)
204:         self.slot_position = nn.Embedding(2, width)
205:         self.norm = RMSNorm(width)
206: 
207:     def forward(self, images, coordinates):
208:         if images.shape != (2, 1, 32, 32) or coordinates.shape != (2, 2):
209:             raise ValueError("Image payload needs [2,1,32,32] cells and [2,2] public x,y centers")
210:         features = F.gelu(self.projection(self.cnn(images).flatten(1)))
211:         logits = self.head(features)
212:         tokens = features + self.symbol_projection(logits.softmax(-1))
213:         tokens = tokens + self.coordinates(coordinates) + self.slot_position.weight
214:         return self.norm(tokens), logits
215: 
216: 
217: class OCREncoder(nn.Module):
218:     def __init__(self, width):
219:         super().__init__()
220:         self.cnn = nn.Sequential(
221:             nn.Conv2d(1, 16, 3, stride=2, padding=1),
222:             nn.GELU(),
223:             nn.Conv2d(16, 32, 3, stride=2, padding=1),
224:             nn.GELU(),
225:             nn.Conv2d(32, 64, 3, padding=1),
226:             nn.GELU(),
227:         )
228:         self.column_projection = nn.Linear(64 * 8, 64)
229:         self.sequence = nn.GRU(64, 48, batch_first=True, bidirectional=True)
230:         self.head = nn.Linear(96, len(OCR_CHARACTERS) + 1)
231:         self.projection = nn.Linear(96, width)
232:         self.symbol_projection = nn.Linear(len(OCR_CHARACTERS) + 1, width, bias=False)
233:         self.column_position = nn.Embedding(32, width)
234:         self.norm = RMSNorm(width)
235: 
236:     def forward(self, image):
237:         if image.shape != (1, 32, 128):
238:             raise ValueError("OCR payload must be an explicit ROI crop [1,32,128]")
239:         features = self.cnn(image[None]).permute(0, 3, 1, 2).flatten(2)
240:         sequence, _ = self.sequence(F.gelu(self.column_projection(features)))
241:         sequence = sequence[0]
242:         logits = self.head(sequence)
243:         tokens = self.projection(sequence) + self.symbol_projection(logits.softmax(-1)) + self.column_position.weight
244:         return self.norm(tokens), logits
245: 
246: 
247: class AudioEncoder(nn.Module):
248:     def __init__(self, width):
249:         super().__init__()
250:         self.temporal = nn.Sequential(
251:             nn.Conv1d(40, 48, 5, stride=2, padding=2),
252:             nn.GELU(),
253:             nn.Conv1d(48, 64, 5, stride=2, padding=2),
254:             nn.GELU(),
255:         )
256:         self.sequence = nn.GRU(64, 48, batch_first=True, bidirectional=True)
257:         self.head = nn.Linear(96, 3)
258:         self.projection = nn.Linear(96, width)
259:         self.symbol_projection = nn.Linear(3, width, bias=False)
260:         self.time_position = nn.Embedding(16, width)
261:         self.norm = RMSNorm(width)
262: 
263:     def forward(self, frames, valid=None):
264:         if frames.ndim != 2 or frames.shape[1] != 40 or frames.shape[0] < 1:
265:             raise ValueError("Audio payload must contain the complete [time,40] log-mel sequence")
266:         if valid is not None:
267:             if valid.dtype != torch.bool or valid.shape != frames.shape[:1]:
268:                 raise ValueError("Audio valid mask must be boolean [time]")
269:             length = int(valid.sum())
270:             expected = torch.arange(len(valid), device=valid.device) < length
271:             if length == 0 or not torch.equal(valid, expected):
272:                 raise ValueError("Audio padding must follow a nonempty, contiguous valid prefix")
273:             # Trim only declared padding, never truncate real sound for a duration limit.
274:             frames = frames[:length]
275:         features = self.temporal(frames.transpose(0, 1)[None]).transpose(1, 2)
276:         sequence, _ = self.sequence(features)
277:         sequence = sequence[0]
278:         logits = self.head(sequence.mean(0))
279:         # Sixteen ordered temporal intervals, rather than one frequency-average token.
280:         temporal = F.adaptive_avg_pool1d(sequence.transpose(0, 1)[None], 16)[0].transpose(0, 1)
281:         tokens = self.projection(temporal) + self.symbol_projection(logits.softmax(-1))[None]
282:         return self.norm(tokens + self.time_position.weight), logits

tiny_perceptron/selftrained/model.py:L296-L419
296:     def __init__(self, config=None):
297:         super().__init__()
298:         self.config = config or SelftrainedConfig()
299:         self.lm = SelftrainedLanguageModel(self.config)
300:         self.vision_encoder = VisionEncoder(self.config.width)
301:         self.ocr_encoder = OCREncoder(self.config.width)
302:         self.audio_encoder = AudioEncoder(self.config.width)
303: 
304:     @staticmethod
305:     def modality_token_count(kind):
306:         return MODALITY_TOKENS[kind]
307: 
308:     def _embeddings(self, input_ids, attention_mask, modalities):
309:         embeddings = self.lm.embedding(input_ids)
310:         modalities = [[] for _ in range(len(input_ids))] if modalities is None else modalities
311:         if len(modalities) != len(input_ids):
312:             raise ValueError("Modalities must have one ordered payload list per batch row")
313:         perception = []
314:         for row, payloads in enumerate(modalities):
315:             locations = {kind: (input_ids[row] == token).nonzero().flatten() for kind, token in MODALITY_IDS.items()}
316:             expected = {
317:                 kind: sum(p.get("kind") == kind for p in payloads) * count for kind, count in MODALITY_TOKENS.items()
318:             }
319:             if any(len(locations[kind]) != expected[kind] for kind in locations):
320:                 raise ValueError(
321:                     "Reserved modality slots must exactly match the payloads; no missing or gold-filled slots"
322:                 )
323:             consumed = dict.fromkeys(MODALITY_TOKENS, 0)
324:             for index, payload in enumerate(payloads):
325:                 if set(payload) - {"kind", "values", "valid", "coordinates"}:
326:                     raise ValueError("Model payloads accept pixels/features and public geometry only")
327:                 kind = payload.get("kind")
328:                 if kind not in MODALITY_TOKENS:
329:                     raise ValueError("Unknown modality payload")
330:                 values = torch.as_tensor(payload["values"], device=embeddings.device).to(embeddings.dtype)
331:                 if not torch.isfinite(values).all():
332:                     raise ValueError("Modality input contains nonfinite values")
333:                 if kind == "image":
334:                     if "coordinates" not in payload:
335:                         raise ValueError("Image payload must supply public slot coordinates")
336:                     coordinates = torch.as_tensor(payload["coordinates"], device=embeddings.device).to(embeddings.dtype)
337:                     if not torch.isfinite(coordinates).all() or not ((coordinates >= 0) & (coordinates <= 1)).all():
338:                         raise ValueError("Image coordinates must be finite normalized x,y centers")
339:                     tokens, logits = self.vision_encoder(values, coordinates)
340:                 elif kind == "ocr":
341:                     tokens, logits = self.ocr_encoder(values)
342:                 else:
343:                     valid = payload.get("valid")
344:                     valid = None if valid is None else torch.as_tensor(valid, device=embeddings.device)
345:                     tokens, logits = self.audio_encoder(values, valid)
346:                 start, count = consumed[kind], MODALITY_TOKENS[kind]
347:                 slots = locations[kind][start : start + count]
348:                 if attention_mask is not None and not attention_mask[row, slots].all():
349:                     raise ValueError("A modality slot cannot be padding")
350:                 if len(slots) > 1 and not torch.equal(slots[1:], slots[:-1] + 1):
351:                     raise ValueError("Each modality payload must occupy one contiguous reserved span")
352:                 embeddings[row, slots] = tokens
353:                 consumed[kind] += count
354:                 perception.append({"batch_index": row, "modality_index": index, "kind": kind, "logits": logits})
355:         return embeddings, perception
356: 
357:     def forward(
358:         self, input_ids, attention_mask=None, labels=None, modalities=None, *, positions=None, segments=None, cache=None
359:     ):
360:         if input_ids.ndim != 2 or input_ids.dtype != torch.long:
361:             raise ValueError("input_ids must be a long [batch,time] tensor")
362:         if cache is None:
363:             embeddings, perception = self._embeddings(input_ids, attention_mask, modalities)
364:         else:
365:             if modalities is not None:
366:                 raise ValueError("Cached continuation uses the encoded prefix; do not resend modalities")
367:             if any((input_ids == token).any() for token in MODALITY_IDS.values()):
368:                 raise ValueError("A cached continuation cannot introduce unencoded modality slots")
369:             embeddings, perception = self.lm.embedding(input_ids), []
370:         result = self.lm(embeddings, attention_mask, positions, segments, cache)
371:         result["perception"] = perception
372:         result["loss"] = None
373:         if labels is not None:
374:             if labels.shape != input_ids.shape:
375:                 raise ValueError("Labels must already be shifted and align with the reserved modality slots")
376:             if attention_mask is not None and (labels[~attention_mask] != IGNORE).any():
377:                 raise ValueError("Padding cannot carry supervised labels")
378:             total, count = loss_sum(result["logits"], labels)
379:             result["language_loss"] = total / count
380:             result["supervised_tokens"] = count
381:             result["loss"] = result["language_loss"] + self.config.auxiliary_weight * result["aux_loss"]
382:         return result
383: 
384:     def generate(self, input_ids, modalities=None, max_new_tokens=96, eos_id=2, use_cache=True, temperature=0.0):
385:         """Return newly generated tokens only; one unpadded conversation per call."""
386:         if input_ids.ndim != 2 or input_ids.shape[0] != 1 or (input_ids == 0).any():
387:             raise ValueError("Generation accepts one unpadded conversation")
388:         if max_new_tokens < 0:
389:             raise ValueError("max_new_tokens cannot be negative")
390:         was_training = self.training
391:         self.eval()
392:         sequence, state, generated = input_ids.clone(), None, []
393:         try:
394:             with torch.no_grad():
395:                 for _ in range(max_new_tokens):
396:                     if sequence.shape[1] >= self.config.max_length:
397:                         break
398:                     current = sequence if state is None else sequence[:, -1:]
399:                     result = self(
400:                         current,
401:                         modalities=modalities if state is None else None,
402:                         cache=state if use_cache else None,
403:                     )
404:                     scores = result["logits"][:, -1].clone()
405:                     # Control markers belong to the renderer. EOS and visible UNK remain available.
406:                     scores[:, [0, 1, 3, 4, 5, 6, 7, 8, 9]] = float("-inf")
407:                     next_id = (
408:                         scores.argmax(-1, keepdim=True)
409:                         if temperature <= 0
410:                         else torch.multinomial((scores / temperature).softmax(-1), 1)
411:                     )
412:                     generated.append(next_id)
413:                     sequence = torch.cat((sequence, next_id), 1)
414:                     state = result["cache"] if use_cache else None
415:                     if next_id.item() == eos_id:
416:                         break
417:         finally:
418:             self.train(was_training)
419:         return torch.cat(generated, 1) if generated else input_ids.new_empty((1, 0))

108: class MaskedBlock(Block):
109:     def __init__(self, language_config, config):
110:         # The reused block supplies the same attention, norms and residual structure.
111:         super().__init__(language_config)
112:         self.ffn = (
113:             PaddingSafeMoE(config.width, config.experts, config.top_k, config.ffn_hidden)
114:             if config.architecture == "moe"
115:             else DenseFFN(config.width, config.ffn_hidden)
116:         )
117: 
118:     def forward(self, x, valid=None, positions=None, segments=None, cache=None, routing_valid=None):
119:         attended, state = self.attention(self.norm1(x), valid, positions, segments, cache)
120:         x = x + attended
121:         h = self.norm2(x)
122:         auxiliary, routing = x.new_zeros(()), None
123:         if isinstance(self.ffn, PaddingSafeMoE):
124:             h, auxiliary, routing = self.ffn(h, routing_valid)
125:         else:
126:             h = self.ffn(h)
127:         x = x + h
128:         if routing_valid is not None:
129:             x = x.masked_fill(~routing_valid[..., None], 0)
130:         return x, state, auxiliary, routing

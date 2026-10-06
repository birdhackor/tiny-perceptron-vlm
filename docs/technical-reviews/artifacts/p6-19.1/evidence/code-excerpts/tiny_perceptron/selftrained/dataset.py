tiny_perceptron/selftrained/dataset.py:L118-L220
118:     def _modality(self, metadata, kind):
119:         key = json.dumps({k: metadata.get(k) for k in ("image", "audio", "roi", "image_layout")}, sort_keys=True)
120:         cache_key = (kind, key)
121:         if cache_key in self.cache:
122:             return self.cache[cache_key]
123:         if kind == "audio":
124:             import soundfile as sf
125: 
126:             audio, sample_rate = sf.read(_asset_path(self.asset_dir, metadata["audio"]), dtype="float32")
127:             if audio.ndim == 2:
128:                 audio = audio.mean(axis=1)
129:             if len(audio) == 0 or sample_rate < 1:
130:                 raise ValueError("音訊為空或 sample rate 無效")
131:             waveform = torch.from_numpy(audio.copy())
132:             # 全錄音，32 ms FFT / 10 ms hop；不以來源轉寫或意圖裁剪。
133:             values = log_mel(waveform, sample_rate, round(sample_rate * 0.032), round(sample_rate * 0.01), 40).T
134:             values = (values - values.mean()) / values.std().clamp(min=1e-5)
135:             payload = {"kind": kind, "values": values, "valid": torch.ones(len(values), dtype=torch.bool)}
136:         else:
137:             with Image.open(_asset_path(self.asset_dir, metadata["image"])) as source:
138:                 image = source.convert("L")
139:                 if kind == "ocr":
140:                     roi = metadata.get("roi")
141:                     if not roi or len(roi) != 4:
142:                         raise ValueError("OCR 需要公開指定的 roi")
143:                     if not (0 <= roi[0] < roi[2] <= image.width and 0 <= roi[1] < roi[3] <= image.height):
144:                         raise ValueError("OCR roi 必須在影像內且有正寬高")
145:                     crop = image.crop(tuple(roi))
146:                     # Public ROI geometry alone determines scale; no target length.
147:                     # Keep glyph aspect ratio and a common 32px cell scale across
148:                     # 1–4-character lines, then leave the right margin white.
149:                     width = max(1, round(crop.width * 32 / crop.height))
150:                     if width > 128:
151:                         raise ValueError("OCR roi 的等比高32寬度超過128")
152:                     resized = crop.resize((width, 32), Image.Resampling.BILINEAR)
153:                     padded = Image.new("L", (128, 32), 255)
154:                     padded.paste(resized, (0, 0))
155:                     values = torch.from_numpy(np.asarray(padded, dtype=np.float32).copy())[None] / 255
156:                     payload = {"kind": kind, "values": values}
157:                 else:
158:                     slots = metadata.get("image_layout", {}).get("slots")
159:                     if slots is None:
160:                         slots = [[0, 0, image.width, image.height], [0, 0, image.width, image.height]]
161:                     if len(slots) != 2:
162:                         raise ValueError("服飾圖需要兩個公開幾何格位")
163:                     crops = [image.crop(tuple(box)).resize((32, 32), Image.Resampling.BILINEAR) for box in slots]
164:                     values = torch.stack(
165:                         [torch.from_numpy(np.asarray(c, dtype=np.float32).copy())[None] / 255 for c in crops]
166:                     )
167:                     coordinates = torch.tensor(
168:                         [[(b[0] + b[2]) / (2 * image.width), (b[1] + b[3]) / (2 * image.height)] for b in slots]
169:                     )
170:                     payload = {"kind": kind, "values": values, "coordinates": coordinates}
171:         self.cache[cache_key] = payload
172:         return payload
173: 
174:     def encode(self, record, *, generation=False, pretrain=False, messages=None, modality_override=None):
175:         tokenizer = self.tokenizer
176:         if messages is None:
177:             messages = record["messages"][:-1] if generation else record["messages"]
178:         ids, labels, modalities = [_special(tokenizer, "bos")], [-100], []
179:         user_indexes = [i for i, m in enumerate(messages) if m["role"] == "user"]
180:         attach_index = record.get("modality_message_index", user_indexes[-1] if user_indexes else -1)
181:         if (record.get("image") or record.get("audio")) and attach_index not in user_indexes:
182:             raise ValueError("公開 modality_message_index 必須指向保留的 user 訊息")
183:         for index, message in enumerate(messages):
184:             ids.append(_special(tokenizer, message["role"]))
185:             labels.append(_special(tokenizer, message["role"]) if pretrain else -100)
186:             metadata = dict(message)
187:             if index == attach_index:
188:                 metadata.update({k: record[k] for k in ("image", "audio", "roi", "image_layout") if k in record})
189:             if (
190:                 index == attach_index
191:                 and record["task"] == "ocr"
192:                 and metadata.get("image")
193:                 and metadata.get("roi") is None
194:             ):
195:                 raise ValueError("目前 OCR 訊息需要公開指定的 roi")
196:             kinds = ["ocr" if metadata.get("roi") is not None else "image"] if metadata.get("image") else []
197:             if metadata.get("audio"):
198:                 kinds.append("audio")
199:             for kind in kinds:
200:                 ids.extend([_special(tokenizer, kind)] * MODALITY_SLOTS[kind])
201:                 labels.extend([-100] * MODALITY_SLOTS[kind])
202:                 payload = self._modality(metadata, kind)
203:                 if modality_override == "zero":
204:                     payload = {**payload, "values": torch.zeros_like(payload["values"])}
205:                 modalities.append(payload)
206:             content_ids = tokenizer.encode(message["content"])
207:             if isinstance(content_ids, torch.Tensor):
208:                 content_ids = content_ids.tolist()
209:             ids.extend(content_ids)
210:             labels.extend(content_ids if message["role"] == "assistant" or pretrain else [-100] * len(content_ids))
211:             ids.append(_special(tokenizer, "eos"))
212:             labels.append(_special(tokenizer, "eos") if message["role"] == "assistant" or pretrain else -100)
213:         if generation:
214:             ids.append(_special(tokenizer, "assistant"))
215:             labels.append(-100)
216:         if len(ids) > self.context:
217:             raise ValueError(f"record {record['id']} 需要 {len(ids)} tokens，超過 context={self.context}；禁止靜默截斷")
218:         # forward 的 labels 已對齊「下一個 token」；assistant role 預測首字。
219:         labels = labels[1:] + [-100]
220:         return {"input_ids": ids, "labels": labels, "modalities": modalities, "record": record}

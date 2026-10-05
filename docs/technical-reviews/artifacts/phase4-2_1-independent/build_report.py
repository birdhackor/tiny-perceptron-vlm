from pathlib import Path
import hashlib
import json
import shlex

dest = Path(__file__).resolve().parent
root = dest.parents[3]
relative = dest.relative_to(root).as_posix()
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
extraction = json.loads((dest / 'original/extraction.json').read_text())
intro = json.loads((dest / 'intro-receipt.json').read_text())
probe = json.loads((dest / 'probe-stdout.txt').read_text())
original_execution = json.loads((dest / 'original/execution.json').read_text())
original_environment = json.loads((dest / 'original/environment.json').read_text())
probe_execution = json.loads((dest / 'probe-execution.json').read_text())
render_execution = json.loads((dest / 'figure-render-execution.json').read_text())
receipts = json.loads((dest / 'sources/receipts.json').read_text())
environment = probe['environment']
artifacts = []

def artifact(identifier, name, kind, description, **extra):
    path = dest / name
    artifacts.append({'id': identifier, 'path': path.relative_to(root).as_posix(),
                      'sha256': digest(path), 'kind': kind, 'description': description, **extra})

artifact('original-run', 'original/stdout.txt', 'execution', '本節原始 fence CPU stdout；一個 fence 實際完成。',
         command=original_execution['command'], result='exit 0；torch.Size([1,3,3])；貓/看/貓三列；equal 斷言通過。',
         environment={key: original_environment[key] for key in ['python', 'torch', 'torch_git_version', 'device_requested', 'cuda_build', 'cuda_available']})
artifact('probe-run', 'probe-stdout.txt', 'execution', '審閱者有界 CPU 變化、梯度與單次合成更新的逐項結果。',
         command=shlex.join(probe_execution['command_argv']), result='exit 0；所有斷言通過；本節結果與必要變化一致。',
         environment=environment)
artifact('render-run', 'figure-render-execution.json', 'execution', '實際 SVG→PNG 匯出命令與完成狀態。',
         command=shlex.join(render_execution['command_argv']), result='exit 0；640×1060 PNG 已產生且審閱者用 view_image 親看。',
         environment={'renderer': 'Inkscape 1.4 (e7c3feb100, 2024-10-09)', 'device': 'CPU', 'view_tool': 'view_image'})
for identifier, name, kind, description in [
    ('section-snapshot', 'original/section.md', 'source_snapshot', '原始 UTF-8 小節 bytes，未正規化換行。'),
    ('intro-snapshot', 'chapter-intro.md', 'source_snapshot', '本章導言原始 UTF-8 bytes。'),
    ('intro-receipt', 'intro-receipt.json', 'source_snapshot', '導言原始位元組指紋、byte 數與讀取行範圍。'),
    ('chapter-snapshot', 'chapter-original.md', 'source_snapshot', '親讀時的第 2 章原始 UTF-8 全檔快照。'),
    ('original-fence', 'original/fence-1.py', 'code', '本節唯一 Python fence 的原始 bytes。'),
    ('bootstrap', 'original/bootstrap.py', 'code', '實際執行的 BOOTSTRAP bytes。'),
    ('bootstrap-source', 'original/build_course.py', 'code', '亲讀並用於擷取 BOOTSTRAP 的實際 helper 原檔。'),
    ('helper-source', 'original/section_facts.py', 'code', '親讀的 section_facts 原始契約與版本。'),
    ('extraction', 'original/extraction.json', 'source_snapshot', '原始小節/fence/圖 SHA、首行與擷取資訊。'),
    ('original-receipt', 'original/execution.json', 'source_snapshot', '原始 fence 真正執行命令、exit、時間、輸出指紋。'),
    ('original-environment', 'original/environment.json', 'source_snapshot', '原始 fence CPU 環境、guard、版本；無 guard 事件。'),
    ('original-stderr', 'original/stderr.txt', 'source_snapshot', '原始 fence stderr 為空。'),
    ('probe-code', 'probe.py', 'code', '獨立有界查證原碼；含有意義變化与一筆一步合成更新。'),
    ('probe-receipt', 'probe-execution.json', 'source_snapshot', 'probe 真正命令、exit、時間與 stdout/stderr SHA。'),
    ('probe-stderr', 'probe-stderr.txt', 'source_snapshot', 'probe stderr 為空。'),
    ('figure-png', 'embedding-purpose.png', 'figure_render', 'Inkscape 1.4 的 PNG；已實際 view_image 親看。'),
    ('figure-stderr', 'figure-render-stderr.txt', 'source_snapshot', '渲染 stderr 的 PangoFT2FontMap/GtkRecentManager 警告。'),
    ('figure-source', 'original/figures/course/figures/rewrite-02-embedding-purpose.svg', 'source_snapshot', '親讀/渲染圖的原始 SVG bytes。'),
    ('acquisition-code', 'acquire_sources.py', 'code', '一手來源 HTTPS 取得原碼；只下載原论文及官方原碼。'),
    ('acquisition-receipts', 'sources/receipts.json', 'source_snapshot', 'HTTPS URL、HTTP 200、日期、bytes 和 SHA-256。'),
    ('reproduce-code', 'preserve_and_run.py', 'code', '永久保存、CPU probe 與 render/PDF 文字擷取的命令原碼。'),
    ('inspection-notes', 'inspection-notes.md', 'derivation', '本審閱者自己的親讀範圍、支持界線、算式與圖親看紀錄。'),
    ('report-builder', 'build_report.py', 'code', '本審閱者獨立建立報告的原碼，不讀舊報告。'),
    ('bengio-text', 'sources/bengio-2003-jmlr.txt', 'source_snapshot', '原 JMLR PDF 的 pdftotext -layout；只用於定位親讀段落。'),
    ('vaswani-text', 'sources/vaswani-2017-v7.txt', 'source_snapshot', 'arXiv v7 原 PDF 的 pdftotext -layout；親讀 §§3.2.3/3.4/3.5。'),
]:
    artifact(identifier, name, kind, description)
for receipt in receipts:
    artifact('raw-' + receipt['name'], 'sources/' + receipt['name'], 'source_snapshot',
             '實際取得與親讀的一手來源原 bytes；版本/URL/receipt 均登記。')
for name in ['bengio-2003-jmlr', 'vaswani-2017-v7']:
    artifact(name + '-extraction-receipt', name + '-pdftotext-execution.json', 'source_snapshot', '實際 pdftotext -layout 命令，exit 0。')

sources = []
def official(identifier, filename, title, kind, version, authority, inspection):
    receipt = next(r for r in receipts if r['name'] == filename)
    sources.append({'id': identifier, 'kind': kind, 'title': title, 'url': receipt['url'],
                    'version': version, 'verified': True, 'checked_original': True,
                    'accessed_on': receipt['accessed_utc'][:10], 'authority_reason': authority,
                    'inspection_note': inspection, 'snapshot_artifact_id': 'raw-' + filename})

official('bengio', 'bengio-2003-jmlr.pdf', 'A Neural Probabilistic Language Model', 'paper',
         'JMLR 3 (2003), 1137–1155，出版社原 PDF', 'JMLR 出版社提供作者原始論文全文。',
         '親讀 Abstract、§1.1 p.1139、§2 pp.1141–1143、Figure 1、Eq.(1) 及梯度更新式：C 是共用 |V|×m 自由參數表，g 產生下一詞機率，兩者共同學習；不引用本論文實測表支持本節人工例子。')
official('vaswani', 'vaswani-2017-v7.pdf', 'Attention Is All You Need', 'paper',
         'arXiv:1706.03762v7', '作者在 arXiv 發布的原始論文固定版本。',
         '親讀 §3.2.3 p.5 的 self-attention 上下文、§3.4 p.5 的 learned token embeddings 與 §3.5 p.6 的 positional encoding；僅支持同字查表起點相同而上下文/位置可能改中間表示。')
official('torch-sparse', 'pytorch-v2.9.0-sparse.py', 'PyTorch nn.Embedding 官方實作与 API 契約', 'official_source',
         'pytorch/pytorch tag v2.9.0', 'PyTorch 官方 GitHub repository 的固定 tag 原始碼。',
         '親讀 lines 15–204：indices 查表、weight 形狀、N(0,1) 初始化、預設可求導 Parameter、reset_parameters/forward。來源是 v2.9.0，實際安裝為 2.14.1+cpu；各自如實記錄。')
official('torch-functional', 'pytorch-v2.9.0-functional.py', 'PyTorch F.embedding 官方 API/轉派', 'official_source',
         'pytorch/pytorch tag v2.9.0', 'PyTorch 官方 GitHub repository 的固定 tag 原始碼。',
         '親讀 lines 2428–2546 的 F.embedding：LongTensor indices、V×D 浮點 weight、附加 D 軸；defaults 无 padding/max_norm/frequency scaling；最終呼叫 torch.embedding。')
official('torch-native', 'pytorch-v2.9.0-Embedding.cpp', 'PyTorch embedding 的 CPU lookup 与 backward', 'official_source',
         'pytorch/pytorch tag v2.9.0', 'PyTorch 官方 ATen CPU 原始實作。',
         '親讀 embedding_symint lines 37–54、embedding_dense_backward_cpu lines 112–178；index_select 取列，梯度先 zero 再依 input ID 累加。只套用本例預設；不延伸至 optimizer weight decay 或 padding。')
official('torch-no-grad', 'pytorch-v2.9.0-grad_mode.py', 'PyTorch no_grad 官方契約', 'official_source',
         'pytorch/pytorch tag v2.9.0', 'PyTorch 官方 autograd 原始實作與 docstring。',
         '親讀 no_grad lines 21–86；區塊內關閉反向求導追蹤，__exit__ 恢復 prior grad-enabled，故手動填表不會永久凍結後續 lookup。')
official('torch-tensor-docs', 'pytorch-v2.9.0-_tensor_docs.py', 'PyTorch Tensor.copy_ 官方文檔原始碼', 'official_source',
         'pytorch/pytorch tag v2.9.0', 'PyTorch 官方 Tensor API docstring 原始檔。',
         '親讀 lines 1204–1224；copy_ 原地從 src 複製元素。本例兩者均為 5×3 float，沒有 broadcast 或 device 差異。')
official('torch-docs', 'pytorch-v2.9.0-_torch_docs.py', 'PyTorch tensor/equal 官方文檔原始碼', 'official_source',
         'pytorch/pytorch tag v2.9.0', 'PyTorch 官方 torch API docstring 原始檔。',
         '親讀 equal lines 4064–4085、tensor lines 9223–9275；equal 核對 shape 與 elements，tensor 從資料建 Tensor。未主張 equal 核對 dtype；本例无 NaN 且 shape/元素一致。')
sources += [
    {'id': 'repo-simple', 'kind': 'repository_code', 'title': 'BigramLM 與 ContextMLP 真正前向契約',
     'path': 'tiny_perceptron/simple.py', 'sha256': digest(root / 'tiny_perceptron/simple.py'),
     'version': '2026-10-05 親讀 working-tree 原始 bytes', 'verified': True,
     'inspection_note': '全檔親讀：BigramLM lines 7–13 的 V×V table/context[:, -1]；ContextMLP lines 16–28 的 V×width lookup 與後續候選 output。probe 真正載入同檔 BigramLM，核對相同末 ID 不受前綴影響。'},
    {'id': 'section-code', 'kind': 'repository_code', 'title': '本節唯一原始 Python fence',
     'path': relative + '/original/fence-1.py', 'sha256': digest(dest / 'original/fence-1.py'),
     'version': extraction['source_sha256'], 'verified': True,
     'inspection_note': '親讀所有21行原碼；Embedding(5,3)、no_grad/copy_ 的人工五列、[1,2,1] lookup、print/equal。没有候選 output、loss、backward 或 step。'},
    {'id': 'derivation', 'kind': 'derivation', 'title': '查表、候選配方与表參數數的獨立推導', 'verified': True,
     'details': 'B=1,C=3,D=3,V=5；[1,2,1] 查出貓/看/貓；score=e·[2,0,1]，2×1+0=2、2×0+0=0、2×1.5+0=3；原貓改首乘數3得3；表參數 V×D=5×3=15。都是無單位示意數字，無統計分母。'},
    {'id': 'execution-original', 'kind': 'execution', 'title': '原碼 fence 的真實 CPU 執行', 'verified': True, 'artifact_id': 'original-run'},
    {'id': 'execution-probe', 'kind': 'execution', 'title': '獨立短 CPU 變化與機制查證', 'verified': True, 'artifact_id': 'probe-run'},
]

def evidence(source, locator, supports):
    return {'source_id': source, 'locator': locator, 'supports': supports}
def verification(expected, observed, details, tolerance=None):
    value = {'method': 'executed', 'expected': expected, 'observed': observed, 'details': details}
    if tolerance:
        value['tolerance'] = tolerance
    return value
def claim(identifier, kind, statement, location, scope, ev, arts, verify=None):
    value = {'id': identifier, 'kind': kind, 'statement': statement, 'location': location,
             'scope': scope, 'status': 'verified', 'evidence': ev, 'artifact_ids': arts}
    if verify:
        value['verification'] = verify
    return value

claims = [
    claim('c1-prior-and-intro', 'software', '上一章接字表直接查 V 個候選分數，只取最後一字；兩個不同前綴同以等號結尾時輸入相同。導言後半是本章方法路線。',
          'course/chapters/02.md:1–7；前置 01.md:202–234', '只核對 BigramLM 輸入依賴，不重審上一章訓練成績，也不宣稱本節已學會較長上下文。',
          [evidence('repo-simple', 'BigramLM lines 7–13', 'table=Embedding(V,V)，forward 只查 contexts[:, -1]。'),
           evidence('execution-probe', 'BigramLM_same_last_id_same_scores / BigramLM_parameter_shape', '实际不同前綴且相同末 ID 的分數 exact equal；table [5,5]。')],
          ['probe-run', 'inspection-notes', 'intro-snapshot'],
          verification('同末 ID 的兩筆前綴有相同候選分數，V=5 table [5,5]。', '兩筆分數 exact equal，table [5,5]。', '真載入 simple.py；構造兩筆長3的 context [0,1,2]、[3,4,2]。')),
    claim('c2-feature-lookup', 'concept', 'embedding 是每字的一列可調實數特徵；ID 作選列約定，數值大小不自帶字義，座標不必有人類語義名称；後續共用配方再算候選。',
          'course/chapters/02.md:7–19', '本例每字指字元 token；人工0/1表只教表示與計算，不保證 learned semantics。',
          [evidence('bengio', '§1.1 p.1139；§2 pp.1141–1142 / Figure 1', 'C(i) 是 m 維實數特徵列，C 有自由參數，C 共用於上下文內詞，後續 g 接表示；不要求特徵坐標有人類名稱。'),
           evidence('torch-sparse', 'Embedding docstring lines 15–44；__init__ lines 166–177', '索引取對應word embeddings；weight 是可調的 num_embeddings×embedding_dim Parameter。'),
           evidence('execution-probe', 'ID_relabeling_preserves_lookup', '重新編號並搬動對應列仍給相同字表示，核對ID只決定選列。')],
          ['probe-run', 'inspection-notes']),
    claim('c3-original-api-result', 'software', 'nn.Embedding(5,3)、no_grad/copy_、torch.tensor 與 torch.equal 在本節按五列人工表查 [1,2,1]，輸出 [1,3,3] 的貓/看/貓，首尾向量相等。',
          'course/chapters/02.md:9–45，Python fence', 'B=1筆、C=3位置、D=3特徵；Python打印为torch.Size([1,3,3])，原碼只查表和核對表示。',
          [evidence('torch-sparse', 'Embedding Shape lines 42–44；forward lines 191–200', '输入 (*), 輸出 (*,D)，5行3维weight與forward查表。'),
           evidence('torch-functional', 'F.embedding lines 2428–2475/2546', '整数索引查浮點矩陣，每輸入位置附加3特徵。'),
           evidence('torch-no-grad', 'no_grad lines 21–86', '填表區塊不記反向求導關係，出區塊恢復追蹤。'),
           evidence('torch-tensor-docs', 'Tensor.copy_ lines 1204–1224', 'copy_ 原地填入指定元素；本例shape完全匹配。'),
           evidence('torch-docs', 'torch.tensor lines 9223–9275；torch.equal lines 4064–4085', '從list建立tensor；equal比較相同尺寸與元素。'),
           evidence('execution-original', 'stdout 的 torch.Size 與 tensor；fence-1.py 末行', '實際原碼輸出三列與equal斷言通过，exit0。')],
          ['original-run', 'original-fence', 'extraction', 'probe-run'],
          verification('[1,3,3]；[[[1,0,0],[0,1,0],[1,0,0]]]；首尾相等。', '原碼 stdout 完全吻合；equal斷言成功；probe逐元素精確核對。', '表與ID字義对照 0。/1貓/2看/3狗/4，完全吻合；不讀取前章另一套ID。')),
    claim('c4-manual-score', 'numeric', '額外手算候選狗配方 2×第一格+第三格，使貓得2、看得0；假設貓首格1→1.5使分數2→3，改後續乘數也改分數。圖與算式一致。',
          'course/chapters/02.md:47–51；rewrite-02-embedding-purpose.svg 下半', '單一候選的示意分數而非機率；配方未加入本節原始程式，不構成訓練或成效證明。',
          [evidence('derivation', 'score=[2,0,1]·e 的四個代入式', '2×1+0=2；2×0+0=0；2×1.5+0=3；改首乘數3使原貓得3。'),
           evidence('execution-probe', 'candidate_scores_original / candidate_scores_after_cat_1_to_1_5 / cat_score_after_recipe_multiplier_2_to_3', 'CPU獨立點積得到[[2,0,2]]、[[3,0,3]]和3，與手算一致。')],
          ['probe-run', 'figure-png', 'render-run', 'inspection-notes'],
          verification('原貓2、看0；假設改貓首格得3，另改乘數得3。', '原[[2,0,2]]，改貓值[[3,0,3]]，改乘數原貓3。', '無單位的toy分數；没有樣本統計分母；已親view圖五列/箭頭/式子和數字。', 'exact equality；所有代入值可在float32精確表示，無四捨五入差。')),
    claim('c5-parameter-count', 'numeric', 'V字、D特徵的表有V×D個可調參數；D可與候選數V不同，本例5×3=15。',
          'course/chapters/02.md:60', '只算embedding表，不包含後續候選層、hidden層或optimizer狀態。',
          [evidence('torch-sparse', 'weight shape lines 39–40；Parameter創建 lines 166–171', '唯一weight形狀(num_embeddings,embedding_dim)，預設可求導。'),
           evidence('bengio', '§2 p.1141 的C定義；p.1143 的參數列舉', 'C有|V|×m自由參數，m与|V|独立；後續候選輸出大小另為|V|。'),
           evidence('derivation', 'V×D=5×3=15', '行數乘列數得15。'),
           evidence('execution-probe', 'table_parameter_count', '實際sum(p.numel())=15，與獨立手算一致。')],
          ['probe-run', 'inspection-notes'],
          verification('只一個5×3 weight，15參數。', 'sum(p.numel())=15；Embedding weight要求梯度。', 'V=5,D=3，D≠V；整數計數無單位、無統計分母。', '整數精確相等。')),
    claim('c6-learning-and-scope', 'concept', '實際訓練以答案代價的梯度回到使用的表列，再由更新步驟改表，可與後續配方共同調整以提高正確下一字機率；建立Embedding本身不會學會關係，預設先從亂數開始。',
          'course/chapters/02.md:51–53、60', '訓練是可用的機制与优化目的；本節人工查表和手算未做loss/backward/step，沒有訓練成功或語言品質結論。使用列敘述限default Embedding lookup的梯度贡献。',
          [evidence('bengio', '§1.1 三步 p.1139；§2 pp.1142–1143，θ=(C,ω)、log-likelihood和梯度更新式', '詞特徵与預測參數共同优化下一詞似然；未出現inputwindow的詞特徵無本次lookup梯度。'),
           evidence('torch-sparse', 'lines 39–40、166–184', 'Embedding預設learnable Parameter，reset_parameters以normal_初始化，不含loss/更新程序。'),
           evidence('torch-native', 'embedding_dense_backward_cpu lines 112–178', '梯度zero初始化并將每个input位置贡献加回该ID的weight列；重複ID累加。'),
           evidence('section-code', '本節原始fence全部', '只有人工copy_/lookup/print/equal；没有候選head、loss、backward或step。'),
           evidence('execution-probe', 'sum_loss_gradient、forward_and_backward_without_step_preserve_table、synthetic_one_step、different_constructor_seeds_produce_different_weights', 'CPU核對反向本身不更新參數；一筆一步的自建head+embedding都更新且目標概率上升；不同seed初值不同。只驗示範機制。')],
          ['probe-run', 'original-fence', 'inspection-notes']),
    claim('c7-exercise', 'software', '把第二個ID由2改1，三位置皆查出[1,0,0]，shape仍為[1,3,3]。',
          'course/chapters/02.md:55 的練習', '只改查表索引，不改变表值、上下文長度或特徵維度。',
          [evidence('torch-sparse', 'Embedding Shape lines 42–44；forward lines 191–200', '每次相同ID取同一weight列，輸出只由inputshape附加D。'),
           evidence('execution-probe', 'exercise_shape / exercise_vectors', '[1,1,1]实跑結果三排[1,0,0]，shape不變。')],
          ['probe-run', 'probe-code'], verification('[1,3,3]且三個[1,0,0]。', '实跑shape[1,3,3]，逐元素exact equal。', '有意义的練習變化：middle ID2→1；資料總筆數1，位置3，特徵3。')),
    claim('c8-contextual-representations', 'concept', '同一字查表起始向量相同；加入鄰字與位置資訊後，各位置的中間表示可能不同。',
          'course/chapters/02.md:55 後半', '可能性而非必然；本節未實作位置或上下文層，本次也沒有據此推論訓練成功。',
          [evidence('torch-sparse', 'Embedding lookup docstring lines 15–20、Shape lines 42–44', '固定表相同ID查同列，不使用額外上下文/位置。'),
           evidence('vaswani', '§3.2.3 p.5；§3.4 p.5；§3.5 p.6', 'self-attention接收邻位置訊息；learned token表示之後加位置編碼，因此同token在不同位置的中间输入与上下文表示可不同。')],
          ['inspection-notes']),
]

report = {
    'schema_version': 1, 'review_stage': 'technical', 'lesson_id': '2.1',
    'source': 'course/chapters/02.md#2.1', 'source_sha256': extraction['source_sha256'],
    'verdict': 'pass', 'reviewer_task': '/root/phase4_factual_coordinator/factual_2_1',
    'reviewer_context': 'fresh', 'author_tasks': [],
    'intro_source': 'course/chapters/02.md:1–4 (raw UTF-8 before first ##)',
    'intro_sha256': intro['intro_sha256'], 'intro_artifact_id': 'intro-snapshot',
    'intro_summary': '上一章只取最近一字，兩個欄位前綴同以等號結尾時無法給不同分配。本章計畫把每字轉成可調向量，按位置合併，再送進可調運算；會區分輸入包含線索與模型是否已學會使用線索。此導言不是本節已驗收訓練品質的聲明。',
    'actual_read_scope': ['course/chapters/02.md:1–200（判定2.1与導言）', 'course/chapters/01.md:202–234、539–548（必要前提）',
                          'tiny_perceptron/simple.py（全檔）', 'docs/review-tools/factual-reviewer-instructions.md（全檔）',
                          'scripts/check_technical_reviews.py（全檔）', 'docs/review-tools/section_facts.py（全檔）',
                          'scripts/build_course.py:26–61 BOOTSTRAP', 'course/figures/rewrite-02-embedding-purpose.svg（全檔原碼與實際PNG親view）',
                          '外部來源親讀範圍逐source.inspection_note與inspection-notes.md記錄'],
    'figure_sha256': extraction['figure_sha256'], 'sources': sources, 'artifacts': artifacts, 'claims': claims,
    'issues': [],
    'checks': {
        'factual_accuracy': {'status': 'pass', 'details': '8組主張逐項查證：前置最後字表、自由特徵/ID約定、原碼API/表示、手算分數、表參數數、梯度/共同學習/初值、練習與上下文表示可能性。無未確定或矛盾主張。', 'claim_ids': [c['id'] for c in claims]},
        'numeric_verification': {'status': 'pass', 'details': 'B1/C3/D3 shape与手動五列表逐元素；score2/0/3；表5×3=15精確計數。無單位toy量、無資料統計分母，exact tolerance。另人工一步只是機制探測，分母/步數已寫明。', 'claim_ids': ['c3-original-api-result', 'c4-manual-score', 'c5-parameter-count', 'c7-exercise']},
        'figure_consistency': {'status': 'pass', 'details': '原 SVG SHA 已記錄；Inkscape1.4实际匯出PNG并view_image亲看。五列ID与数值、lookup→候選配方箭頭、score2/0、改值後3與原文一致；手動/未訓練文字可見，沒有裁切。', 'claim_ids': ['c2-feature-lookup', 'c4-manual-score', 'c6-learning-and-scope']},
        'source_verification': {'status': 'pass', 'details': '自己取得并親讀JMLR原論文、arXiv原論文v7、PyTorch官方v2.9.0原碼，逐來源保留URL、版本、存取日、HTTP200、SHA、locator與自己supports。未讀舊報告；官方源码版本与实装2.14.1+cpu分開記。', 'claim_ids': [c['id'] for c in claims]},
        'limitations': {'status': 'pass', 'details': '本節只手動填表/查表/斷言；候選配方額外手算，未建立輸出層、softmax/loss/更新。原文明确没有訓練成功證明。reviewer人工1樣本/5候選/1步只檢查梯度更新链，未跑完整訓練/GPU或資料模型下載；静態SVG以Inkscape亲看，有初始化警告但圖可判讀，未声称浏览器驗證。', 'claim_ids': ['c1-prior-and-intro', 'c2-feature-lookup', 'c4-manual-score', 'c5-parameter-count', 'c6-learning-and-scope', 'c8-contextual-representations']},
    },
}
report_path = root / 'docs/technical-reviews/2.1.json'
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'report': str(report_path), 'claims': len(claims), 'artifacts': len(artifacts),
                  'section_sha256': report['source_sha256'], 'intro_sha256': report['intro_sha256'], 'verdict': report['verdict']}, ensure_ascii=False))

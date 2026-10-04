"""Serialize this owner's completed independent whole-DATA factual review."""
from pathlib import Path
import hashlib,json,datetime

ROOT=Path.cwd(); P=Path('docs/technical-reviews/artifacts/natural-v4-supplemental/data'); B=Path('outputs/natural-v4/factual-research/data')
DOC='docs/natural-assistant/v4/DATA.md'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(DOC)=='d34a957f2679ad6134fedfa919a4298e59ab9fc511e4fce081bf5490212d4072'
assert Path(DOC).read_bytes()==(P/'DATA.current.fullfile.md').read_bytes()==(B/'preview-DATA.raw.md').read_bytes()
cpu=json.loads((P/'cpu-audit-result.json').read_bytes())
assert not (P/'cpu-audit.stderr.txt').read_bytes()
browser_events=[record for record in json.loads((P/'browser-receipt.json').read_bytes()) if 'action' in record]
assert not (P/'browser.stderr.txt').read_bytes()
retrieval={}
for name in ['authority-retrieval.json','authority-followup.json','authority-final-fetch.json','versioned-authority-retrieval.json','apache-retrieval.json']:
    for record in json.loads((P/name).read_bytes()):
        retrieval[record['name']]=record
sources=[]; artifacts=[]; claims=[]
def official(id,name,title,version,note,reason,kind='official_docs'):
    r=retrieval[name]
    assert r.get('status')==200 and sha(B/name)==r['sha256']
    sources.append({'id':id,'kind':kind,'title':title,'url':r['url'],'version':version,'verified':True,
                    'checked_original':True,'accessed_on':'2026-10-04','authority_reason':reason,'inspection_note':note,
                    'retrieval_sha256':r['sha256'],'retrieved_bytes':r['bytes'],'ignored_original_path':str(B/name)})
def repo(id,path,title,note,version='current bytes inspected 2026-10-04'):
    sources.append({'id':id,'kind':'repository_code','title':title,'path':path,'sha256':sha(path),'version':version,
                    'verified':True,'inspection_note':note})
def artifact(id,name,kind,description,**extra):
    path=P/name
    artifacts.append({'id':id,'kind':kind,'path':str(path),'sha256':sha(path),'description':description,**extra})
def evidence(id,locator,supports):return {'source_id':id,'locator':locator,'supports':supports}
def claim(id,kind,statement,location,evidence_list,artifact_ids,scope,verification=None):
    c={'id':id,'kind':kind,'statement':statement,'location':location,'status':'verified','evidence':evidence_list,
       'artifact_ids':artifact_ids,'scope':scope}
    if verification is not None:c['verification']=verification
    claims.append(c)

official('lora','lora-paper.pdf','LoRA: Low-Rank Adaptation of Large Language Models','arXiv:2106.09685v2, 2021-10-16',
         'Read abstract, introduction, section 4.1 equation (3), deployment paragraph and section 4.2 footnote 4: frozen W0 remains needed, A/B are small updates and may be merged. The paper does not guarantee this Qwen3-VL course adaptation improves quality.',
         'Original LoRA authors paper on arXiv.',kind='paper')
official('docci','docci-paper.html','DOCCI: Descriptions of Connected and Contrasting Images','arXiv:2404.19753v1 HTML, 2024-04-30',
         'Read original dataset construction/annotation discussion, Appendix C.1/C.2 (149 image-similarity clusters, human staged annotation), Appendix F.2 (k-means cluster IDs, not manually established sessions), and F licensing. Read necessary original prose, not a candidate-source summary.',
         'Original dataset authors paper hosted by arXiv.',kind='paper')
official('docci_card','docci-card.md','Google DOCCI dataset card','a0a43eaf34676ffd008fb6565dd8c2ba00d09100',
         'Read Dataset Summary, Languages, Source Data/Initial Data Collection, Annotation process, Licensing Information: English detailed human descriptions; author/family photographs; CC BY4.0.',
         'Google publisher dataset card at exact selected dataset revision.')
official('docci_site','docci-site.html','DOCCI official project site','retrieved 2026-10-04, complete response SHA recorded',
         'Read Abstract, Downloads and Metadata: detailed image descriptions, 7.1GB full image archive distinct from this thumbnail selection, cluster_id description, Google LLC licensing both annotations and images under CC BY4.0.',
         'Dataset publishers Google research project website.')
official('nvidia','nvidia-card.md','NVIDIA OCR-Synthetic-Multilingual-v1 dataset card','69696a1cc543ef3a0f8e9892a89c17293e915263',
         'Read Dataset Description/License, HDF5 schema, Annotation JSON Schema and zh_hant folder: synthetically generated document images, images/annotations/line_bboxes/full-page labels, explicit reading-order relation_graph, dataset CC BY4.0. No natural street-photo claim.',
         'NVIDIA publisher card for the exact selected source revision.')
official('oasst','oasst2-card.md','OpenAssistant Conversations Dataset Release2','179dd21fc55192153d94adb0e0ce8f69e222bf75',
         'Read YAML license and features, Dataset Structure and message/tree JSON examples, Ready For Export Trees, Using HF Datasets: messages alternate roles; parent_id/message_id/tree_id allow complete path reconstruction; source rank and synthetic fields are metadata rather than correctness certification.',
         'Original OpenAssistant/LAION publisher dataset card at fixed revision.')
official('fleurs','fleurs-card.md','Google FLEURS dataset card','d7c758a6dceecd54a98cac43404d3d576e721f07',
         'Read Dataset Description, Data Fields, Dataset Creation, Other Known Limitations and licensing. Corpus contains human read speech and transcripts; train speakers differ from dev/test; individual selected speaker identities are not supplied; read-speech production mismatch is explicit. No source assistant answers.',
         'Original Google FLEURS publisher card at exact revision.')
official('aishell','aishell-card.md','AISHELL official Hugging Face card','bbe295d530192a4cd41644b711c9aecd087df653',
         'Read complete short card: Apache-2.0 metadata, 400 recording participants, controlled quiet indoor Mandarin recording and professional manual transcription.',
         'AISHELL publisher corpus card at exact audio archive revision.')
official('aishell_openslr','aishell-openslr.html','OpenSLR33 AISHELL-1 corpus listing','SLR33 page retrieved 2026-10-04',
         'Read full resource description and license field. Apache License v2.0 applies to speech data/transcripts; publisher Beijing Shell Shell Technology; controlled microphone recordings downsampled to16kHz. This supports human recording domain, not spontaneous assistant conversations.',
         'Official corpus release catalog maintained by OpenSLR.')
official('cc4','cc-by-4.html','Creative Commons Attribution4.0 legal code','CC BY4.0 International, English legal code',
         'Read section2 grant and section3(a)(1): retain supplied creator/copyright/license/source notices and indicate modifications, include license text or URI. Hashes are not evidence of label correctness.',
         'License issuer original legal code.')
official('cc3','cc-by-3.html','Creative Commons Attribution3.0 legal code','CC BY3.0 Unported, English legal code',
         'Read section3 grant and section4 attribution/restrictions: supplied author/title/source URI and adapted-work credit are retained. This is distinct from MIT.',
         'License issuer original legal code.')
official('cc0','cc0.html','Creative Commons CC0 legal code','CC0 1.0 Universal, English legal code',
         'Read sections1-3 waiver and fallback. CC0 is a waiver with fallback, not the MIT license; no claim that every Commons image has one uniform license.',
         'License issuer original legal code.')
official('apache','apache-license.txt','Apache License2.0','Version2.0, January2004',
         'Read section4 Redistribution (license copy, modified-file notices, retained attribution and upstream NOTICE where present). Checked local chat-NOTICE and voice Apache license inclusion in manifest.',
         'Apache Software Foundation original license.')
official('cer','cer-metric-pinned.md','Hugging Face Evaluate CER metric card','51383b3dc3730ff33f437e8a1388e85ae38060af',
         'Read Metric description formula CER=(S+D+I)/N and N=reference characters; insertion-heavy CER may exceed1. Dataset normalization must be specified separately. Empty reference division is mathematically undefined, with current project None checked on CPU.',
         'Metric maintainers original versioned metric definition, corroborated by transparent edit-distance derivation.',kind='official_source')
official('evaluation','sklearn-evaluation.html','scikit-learn cross-validation guide','scikit-learn1.7 documentation',
         'Read section3.1 introduction on train/validation/final test and section3.1.2.4 grouped dependent data/GroupKFold: keep an entire dependency group out of train when evaluating unseen groups; repeated test tuning leaks information.',
         'Original evaluation library maintainers authoritative guide.')
official('leakage','sklearn-pitfalls.html','scikit-learn Common pitfalls','scikit-learn1.7 documentation',
         'Read section11.2/11.2.1: test information must not influence model choices, split before learned preparation; source families rather than derivative row counts are an application-specific dependency unit.',
         'Original maintainers evaluation methodology guide.')
official('venv','python-venv.html','Python venv documentation','Python3.13.16 docs, reviewed against existing Python3.13.5 CLI scope',
         'Read module introduction and creation command: venv creates an isolated directory interpreter/package environment; requires an existing base Python and venv support. No installation or new environment created by this review.',
         'Python language original standard-library documentation.')
official('git_clone','git-clone.html','Git clone documentation','official online git-clone manual retrieved2026-10-04',
         'Read --no-checkout option: skips checkout of HEAD after clone; a separate checkout of the fixed revision is therefore needed. Commands were source parsed, never executed.',
         'Git project original command manual.')
official('lfs_spec','git-lfs-pointer-pinned.md','Git LFS Specification: The Pointer','0043a645047926f4bd7f7091299095528253d575',
         'Read The Pointer section: repository contains small UTF8 pointer, oid SHA256 and full-object byte size. A pointer is not compressed dataset bytes; own small archive test rejects pointer replacement.',
         'Original Git LFS implementation repository specification.',kind='official_source')
official('lfs_config','git-lfs-config-pinned.md','git-lfs-config official manual','0043a645047926f4bd7f7091299095528253d575',
         'Read GIT_LFS_SKIP_SMUDGE environment-variable entry: value1 skips pointer-to-object conversion at checkout in smudge/filter-process.',
         'Original Git LFS command manual source.',kind='official_source')
for key,name in [('pillow','pillow-12.3.0'),('h5py','h5py-3.16.0'),('numpy','numpy-2.5.3'),('soundfile','soundfile-0.13.1')]:
    official('pkg_'+key,name+'.json',key+' official PyPI release metadata',name,
             'Read info.name/version/summary/requires_python and urls[].filename. Exact pinned release exists and has Python3.13-compatible Linux x86_64 wheel (SoundFile generic Python3 wheel); packages respectively handle images, HDF5, arrays and audio. Wheel availability is not an install or full-reconstruction execution claim.',
             'Package maintainers published PyPI release metadata.')
repo('manifest','docs/natural-assistant/v4/manifest.json','Current fixed v4 integrated data manifest',
     'Read/parse all records, source declarations, selected file list and archive list. Independently reconstructed question categories, image/original-page/tree/speaker/sentence units, split overlap and bytes; pinned GitHub9a61ecf manifest retrieved at identical SHA. Embedded data provenance was inspected as records; prior reviewer judgments were not adopted.')
repo('fetch','scripts/fetch_natural_data.py','Natural-data pinned download/verification implementation',
     'Complete357-line source read: validate_manifest/load_manifest/summary/download_checked/extract_checked/verify_directory/fetch_data/main. Student59a1eda original bytes retrieved and identical to current. Offline mock fixtures and CLI help executed; no real dataset fetch.',version='student59a1eda4ed7b6e8609892ec2b9013c821ac93e69; current identical')
repo('build','scripts/build_natural_v4_assets.py','Fixed v4 asset assembly/reconstruction implementation',
     'Complete493-line source read: bounded photos; exact OCR helper and voice replay; assemble family checks; validate_assets hash/split binding; license notices; selected deterministic packaging; exact expected-manifest comparison before final manifest rename. GitHub9a61ecf source bytes are identical.',version='9a61ecf524c9518f33f1501c28aa72997d4a82d0; current identical')
repo('ocr_helper','scripts/prepare_natural_v4_ocr.py','Bounded source OCR and crop reconstruction helper',
     'Complete current313-line source read; read pinned9a61ecf RangeFile/rebuild contracts. Current/pinned file bytes differ only in formatting, independently compared equal Python AST. Exact pinned module was separately imported and rejected fake200 range response. Source hashes are separately retained; no hash substitution.')
repo('voice_helper','scripts/replay_natural_v4_voice.py','Fixed original-WAV replay helper',
     'Read fixed corpus URL/row validation, BoundedReader, WAV validation and replay_archive complete retry/member loop, CLI. FLEURS compressed prefix cap32MiB/archive, AISHELL48MiB/archive across retries. Break after selected members; unread tail not hashed. Exact9a61ecf source identical.')
repo('chat_helper','scripts/prepare_natural_v4_chat.py','Conversation-to-training-row expansion',
     'Complete181-line source read. Alternating messages require complete even user/assistant paths; each assistant turn creates a row with preceding messages as history and unchanged family/split. Original/course sources and notices remain distinct; later sampling does not manufacture new families.')
repo('runtime','tiny_perceptron/natural_assistant.py','Current manifest loading and training/evaluation routing',
     'Read lines1-115 (data-root and file hashes),263-311 (base loading and q/v-only LoRA),520-542 (training only manifest rows train),855-896 (actual ASR transcription metrics),912-939 (actual vs source-text chat). CPU load_manifest validated2371 rows/38audio/1506physical assets. No model loaded or inference performed.')
repo('math','tiny_perceptron/natural_concepts.py','Unicode edit-distance and CER implementation',
     'Read edit_distance/text_error_report lines47-67: Levenshtein DP with unit insertion/deletion/substitution on Unicode code points, reference length denominator, empty reference None. Three actual CPU probes verified5/9/0-character reference boundaries.')
repo('selection','docs/natural-assistant/v4/selection.json','Recorded v4 selection',
     'Complete JSON read: current manifest/protocol/scoring binding and selected_variant base. Audited against recomputed recorded scores and public-release configuration; not accepted as an independent quality review.')
repo('scores','docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json','Original recorded v4 validation counts/gates',
     'Complete JSON read; own CPU reweighted counts, recomputed primary numerator and eligibility gates. Data records retain132 generations per candidate and16ASR recording denominators. This audits prior project records, not this reviewer performing GPU evaluation or regrading answers.')
repo('release','docs/natural-assistant/v4/public-release.json','Current selected public configuration',
     'Complete JSON read: selected_variant base, adapter_parameters0, Qwen3-VL and Whisper-turbo fixed revisions, current data manifest hash. No read of approval/reviewer conclusions or assertion of fresh model execution.')
repo('original_captions','docs/natural-assistant/evidence/v4-research/vision/sources/docci-descriptions.jsonl','Preserved original DOCCI publisher description records',
     'Existing full publisher JSONL bytes SHA independently calculated; all1088 current photo questions matched their full original description/support strings. Personally read complete train_01827 original description and compared newly retrieved/viewed publisher thumbnail. Scope is preserved-source record audit; full original dataset was not redownloaded.',version='Google generation1714384012999810, preserved exact original JSONL SHA c9df481...19800')
sources.append({'id':'commons','kind':'official_docs','title':'Commons official per-file source/author/license metadata',
                'url':'https://commons.wikimedia.org/w/api.php','version':'70 original page IDs, official imageinfo extmetadata snapshots retrieved2026-10-04',
                'verified':True,'checked_original':True,'accessed_on':'2026-10-04',
                'authority_reason':'Wikimedia Commons original file pages and official MediaWiki API, keyed by each physical source page ID.',
                'inspection_note':'Personally read all70 page_id/title/artist/license projections from actual official responses. Each stored original SHA1 and license agrees. Response locators query.pages[page_id].imageinfo[0].extmetadata.Artist/LicenseShortName/LicenseUrl; full original responses remain ignored. Two retrieval SHA receipts and per-file URLs are durable.',
                'retrieval_sha256':[r['sha256'] for r in json.loads((P/'commons-retrieval.json').read_bytes())]})
sources.append({'id':'information','kind':'derivation','title':'Deterministic crop/resize information boundary','verified':True,
                'details':'A crop or interpolation is a deterministic function f(x) of the already available pixel array x. Distinct original scenes can map to the same thumbnail x. Any later deterministic enlargement g(x) is identical for these scenes and therefore cannot establish which absent fine detail belonged to the source. A model may infer a plausible detail from priors, but that is not visible evidence. This supports the cautious DATA claim, not a blanket claim that every visual detail remains unreadable.'})
sources.append({'id':'cpu','kind':'execution','title':'This reviewer independent offline CPU audit','artifact_id':'cpu_stdout','verified':True})
sources.append({'id':'browser','kind':'execution','title':'This reviewer actual Chromium navigation and visible-page inspection','artifact_id':'browser_execution','verified':True})

artifact('fullfile','DATA.current.fullfile.md','source_snapshot','Exact unnormalized UTF8 assigned fullfile, read line1 through EOF169.')
artifact('read_receipt','source-read-receipt.json','source_snapshot','Own assigned-document full read, identity and current-byte receipt.')
artifact('preq','prerequisites.current.excerpts.md','source_snapshot','Actually read current20.4/20.5/20.8-20.12 necessary source excerpts, with original bytes.')
artifact('cpu_code','cpu_audit.py','code','Own bounded offline audit: actual manifest, counts, every selected local hash, audio metadata, extraction/conflict/range caps, CER, recorded scores and provenance.')
artifact('cpu_stdout','cpu-audit.stdout.txt','execution','Actual offline CPU execution output; includes expected/observed comparison and current environment.',
         command='.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/data/cpu_audit.py > docs/technical-reviews/artifacts/natural-v4-supplemental/data/cpu-audit.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-supplemental/data/cpu-audit.stderr.txt',
         result='Completed exit0, empty stderr; exact counts, all1513 declared local files,38WAV metadata, manifest assembly,11 bounded behavior checks, actual-source paired-route stub probe, CER and recorded eligibility assertions verified.',
         environment={'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','Pillow':'12.3.0','numpy':'2.5.3','soundfile':'0.14.0','h5py':'not installed'})
artifact('cpu_result','cpu-audit-result.json','source_snapshot','Detailed own computed result including individual1513 file checks,38audio frame counts, denominators and exact pinned OCR AST/probe.')
artifact('cpu_stderr','cpu-audit.stderr.txt','source_snapshot','Actual final CPU run stderr, empty.')
artifact('browser_code','browser_probe.py','code','Own executed Playwright navigation and visible-text/screenshot capture source; screenshots separately personally inspected.')
artifact('browser_execution','browser.stdout.txt','execution','Actual existing Chromium execution output; full browser receipt includes incidental navigation inventory, not a claim to have reviewed all links.',
         command='.venv/bin/python docs/technical-reviews/artifacts/natural-v4-supplemental/data/browser_probe.py > docs/technical-reviews/artifacts/natural-v4-supplemental/data/browser.stdout.txt 2> docs/technical-reviews/artifacts/natural-v4-supplemental/data/browser.stderr.txt',
         result='Completed exit0, empty stderr; DATA visible assertions true and actual three links clicked to20.4,20.11 and pinned307c325 AISHELL source. Four actual screenshots and visible-text files captured and personally read/viewed.',
         environment={'python':'3.13.5','chromium':'151.0.7922.173 built on Debian GNU/Linux13 (trixie)','device':'cpu','viewport':[1280,900]})
artifact('round2_failure','round2-inspection-error.txt','source_snapshot','Own actual round2 receipt-inspection list/dict error and correction, preserved separately from unchanged round1 failure evidence.')
artifact('additional','additional-audit.json','source_snapshot','Own executed runtime-loader/CLI-help and official package metadata projection; complementary fields audited with durable main CPU source.')
for id,name,description in [('retrieval','authority-retrieval.json','First actual original-authority retrievals including preserved DOCCI PDF bound failure'),('followup','authority-followup.json','Successful original arXiv HTML and official code followups'),('final_fetch','authority-final-fetch.json','Actual preview fullfile/pinned manifest/source and sklearn retrieval SHA receipts'),('versioned','versioned-authority-retrieval.json','Exact official GitLFS and CER commit receipts'),('apache_receipt','apache-retrieval.json','Official Apache license retrieval receipt'),('commons_receipt','commons-retrieval.json','Two original Commons API requests/status/SHA receipts'),('commons_rows','commons-license-audit.json','Own small70-file source/creator/license audit projections'),('heads','archive-head-receipt.json','Three actual anonymous HEAD200 responses with exact content lengths; no archive bodies read'),('cat_receipt','cat-source-retrieval.json','Personally retrieved fixed publisher thumbnail receipt'),('cat_caption','cat-original-caption.audit.txt','One complete preserved official original English caption read; all records verified separately'),('pinned_probe','pinned-ocr-probe.json','Current/pinned SHA and actual exact-pinned Range rejection'),('format_diff','pinned-ocr-format-only.diff.txt','Actual current/pinned code diff; AST equivalence supports scope'),('view_notes','personal-view-notes.txt','Own personal visual interpretations and browser-reading boundaries'),('render_receipt','figure-render-receipt.json','Actual Inkscape commands, exit status, original/render SHAs and harmless font warnings'),('browser_receipt','browser-receipt.json','Actual Chromium goto/click/visible-result record'),('browser_text','browser-DATA.visible-text.txt','Actual browser visible DATA content; source reading remains separate'),('failures','initial-errors-preserved.txt','Own actual exploratory failures and their bounded resolutions; not silently erased')]:
    artifact(id,name,'source_snapshot',description)
for n in ['natural-v4-training-cat','natural-v4-family-crops','natural_photo_evidence','natural_reading_order','natural-v4-asr-two-routes']:
    artifact('fig_'+n,n+'.render.png','figure_render','Personally Inkscape-rendered and view_image-inspected necessary prerequisite SVG; interpretation in personal-view-notes.txt.')
for n in ['browser-DATA.top','browser-link-cat','browser-link-order','browser-link-aishell']:
    artifact('screen_'+n,n+'.png','figure_render','Personally captured in actual Chromium and personally viewed through view_image; URL/click/visible text in browser receipt.')

claim('c01','concept','LoRA updates a small parameter correction while retaining the pretrained base, and must be assessed with that base.','DATA.md lines5-7',
      [evidence('lora','section4.1 equation(3), section4.2 footnote4','Frozen base plus low-rank update, complete pretrained model still used at deployment.'),evidence('runtime','load_core lines263-311','Current implementation loads complete Qwen3-VL and selects language q/v LoRA tensors.')],['preq'],
      'Parameter-efficient adaptation is not a guarantee of higher validation quality; no LoRA/base model was loaded by this reviewer.')
claim('c02','empirical','This version retained the original Qwen3-VL base after validation; the two LoRA candidates did not satisfy the declared acceptance rule.','DATA.md line5; linked20.8',
      [evidence('scores','variants.{base,adapter-step-001039,adapter-step-002077}; integer_weights; denominators','Recorded counts and gate inputs.'),evidence('selection','selected_variant; validation_scoring_sha256','Current selected base binding.'),evidence('release','selected_variant; adapter_parameters; asr_model','Delivered configuration has no adapter and fixed Whisper-turbo.')],['cpu_code','cpu_stdout','cpu_result'],
      'Own audit of original recorded grades/gates only; not own GPU reproduction, independent answer regrading or broad population quality estimate.',
      {'method':'executed','expected':'Recorded primary numerator53277/52285/52705 over75600, both adapter eligibility false, selected base.',
       'observed':cpu['validation_record_audit'],'tolerance':'Exact integer arithmetic; no rounding used for selection.',
       'details':'Sum correct_counts[k]*integer_weights[k]; test each actual>=minimum, completion flag and strict primary improvement. Both typed-reference and actual-ASR chat gates fail for both candidates.',
       'denominators':cpu['validation_record_audit']['denominators']})
claim('c03','concept','Source families are the dependency units: derivatives remain in one split; train updates, validation selects settings, final test is held until configuration freeze. Previously selected-on questions become regression evidence.','DATA.md lines11-19,167',
      [evidence('evaluation','section3.1 introduction;3.1.2.4 Grouped data/GroupKFold','Dependent samples and test tuning must be handled separately.'),evidence('leakage','11.2/11.2.1','Test data must not inform model choices.'),evidence('docci','Appendix C.2 and F.2','DOCCI similarity clusters are k-means groupings, not sessions/photographer guarantees.'),evidence('manifest','split_policy; all rows/audio_rows family and split fields','Course family mapping is explicit and no family or selected physical hash crosses splits.')],['cpu_stdout','cpu_result','fig_natural-v4-family-crops'],
      'This protects the present course selection only; public foundation pretraining overlap, independent capture sessions/photographers and FLEURS dev/test individual speaker disjointness are not established.')
expected={'DOCCI scene':[439,28,42],'DOCCI fact':[439,56,84],'NVIDIA single':[710,0,0],'NVIDIA multiline':[117,0,0],'Commons transcription':[141,10,10],'Commons order':[6,3,3],'Chinese presence':[103,18,18],'OpenAssistant chat':[42,9,13],'course chat':[80,0,0]}
observed={}
for name,key in [('DOCCI scene','DOCCI scene'),('NVIDIA single','NVIDIA ocr'),('NVIDIA multiline','NVIDIA ocr_order'),('Commons transcription','Commons ocr'),('Commons order','Commons ocr_order'),('Chinese presence','Chinese presence'),('OpenAssistant chat','OpenAssistant chat'),('course chat','course chat')]:observed[name]=[cpu['counts'][s].get(key,0) for s in ['train','validation','test']]
observed['DOCCI fact']=[sum(v for k,v in cpu['counts'][s].items() if k.startswith('DOCCI ') and k!='DOCCI scene') for s in ['train','validation','test']]
assert observed==expected
claim('c04','numeric','All table question counts and separately stated physical-source/family counts agree with the fixed manifest.','DATA.md lines21-42',
      [evidence('manifest','rows/audio_rows; sources[].metadata; archives[].files','Complete fixed declarations, separately counted by question, physical file/original-page and family.')],['cpu_code','cpu_stdout','cpu_result'],
      'Question count is not independent-photo, source-page, conversation or recording count; the same photo can support scene/fact/presence and paired voice answers share one recording.',
      {'method':'executed','expected':expected,'observed':observed,'tolerance':'Exact integer equality.',
       'details':'Count task and qa_type for all2371 text/image rows. Unique DOCCI image/family sets yield439/28/42 and126/11/12; retained NVIDIA crops827 on153 pages; Commons70 originals split50/10/10 plus62train crops, families46/10/10. Source-tree chat42/9/13 on40/8/12 trees; authored80rows/50families. Audio38split0/16/22.',
       'denominators':{'text_image_questions':2371,'DOCCI_source_photos':509,'NVIDIA_retained_crop_files':827,'NVIDIA_retained_original_pages':153,'Commons_source_photos':70,'Commons_train_crop_files':62,'source_chat_trees':60,'authored_chat_families':50,'selected_recordings':38,'train_questions':2077,'validation_questions':124,'test_questions':170}})
claim('c05','concept','DOCCI supplies human English detailed image descriptions; the course uses fixed publisher thumbnails and AI-authored Traditional-Chinese adaptations while preserving original provenance.','DATA.md lines46-54',
      [evidence('docci_card','Dataset Summary/Languages; Source Data/Annotation process; Licensing','Original English/human source and photo creators.'),evidence('docci_site','Downloads; license paragraph','Publisher descriptions/images and Google LLC CC BY4.0.'),evidence('original_captions','example_id train_01827 and all selected descriptions','Existing complete original descriptions matched current course records.'),evidence('manifest','photo row references/source fields','Full originals, support sentences, fingerprint and author-view records preserved.')],['cpu_stdout','cpu_result','cat_caption','cat_receipt'],
      'Own audit verified1088 row provenance/caption/support/view-record fields and gallery hashes. Historical viewing is supported by records, not re-performed by this reviewer; only the example cat photograph was personally visually rechecked. No official human Chinese gold or full high-resolution archive retrieval claim.')
claim('c06','concept','The cat example supports two cats on a desk and the left cat raising a front paw, while intent/owner/backstory are not visible evidence. Cropping/enlarging cannot establish details absent from the thumbnail.','DATA.md lines48,52',
      [evidence('original_captions','train_01827 complete description','Two cats; left black-white cat seated with paw over on-screen mouse, right cat standing.'),evidence('information','f(x)/g(x) information boundary','Deterministic crop/enlargement cannot distinguish original scenes giving the same source pixels.')],['cat_receipt','cat_caption','fig_natural-v4-training-cat','view_notes'],
      'Personally viewed exact fixed-generation publisher JPEG and rendered example. A plausible inferred detail is not source-visible ground truth; no assertion that all thumbnail regions are equally readable.')
claim('c07','concept','NVIDIA examples are synthetic Traditional-Chinese documents with source line text/boxes; Commons photos supply separately evaluated real scene text. Crop boundaries and specified reading order determine the task.','DATA.md lines58-64',
      [evidence('nvidia','Dataset Description; zh_hant; HDF5 Format/Annotation JSON Schema/Bounding Box Levels','Synthetic source documents, line text coordinates and reading-order graph.'),evidence('ocr_helper','rebuild original-source dimensions/crop_xyxy/PNG checks','Fixed original-page and crop linkage.'),evidence('manifest','sources metadata ocr-sources.json; rows.references','Separate synthetic/Commons provenance and ordered vs single-line task policies.')],['cpu_result','fig_natural_reading_order','view_notes'],
      'The guide correctly requires geometry and semantic boundary checks; this review verifies source metadata and selected hashes, not independent visual regrading of all827 synthetic crops or a complete-page layout benchmark.')
claim('c08','concept','DOCCI and NVIDIA use CC BY4.0; Commons licenses and creators are retained per individual image, and these materials are not uniformly MIT.','DATA.md lines54,62-64,128',
      [evidence('docci_site','Downloads licensing paragraph','Google LLC images/annotations license.'),evidence('nvidia','License/Terms of Use','NVIDIA dataset CC BY4.0.'),evidence('commons','query.pages[page_id].imageinfo[0].extmetadata Artist/LicenseShortName/LicenseUrl for all70 pages','Each individual source license/artist agrees with course source record.'),evidence('cc4','section3(a)(1)','Attribution, source/license reference, indicate changes.'),evidence('cc3','section4(b)','Author/title/source and adapted-use credit.'),evidence('cc0','sections2-3','Waiver/fallback distinct from MIT.')],['commons_receipt','commons_rows','cpu_result'],
      'Current selected source-license records verified; no inference that unrelated Commons works share these permissions, and no legal determination beyond publisher-specified license scope.')
claim('c09','numeric','CER uses minimal insertion/deletion/substitution count divided by reference Unicode-character count; empty reference has no valid denominator and presence/absence is a separate criterion.','DATA.md line66; necessary20.10/20.12 examples',
      [evidence('cer','Metric description formula CER=(S+D+I)/N','Reference-character denominator.'),evidence('math','edit_distance/text_error_report lines47-67','Current DP implementation and None for empty reference.')],['cpu_stdout','cpu_result','fig_natural_reading_order'],
      'Exact/normalized text and explicit line order are distinct policies. NFKC does not by itself equate 臺/台 or Simplified/Traditional script; CER is not semantic response quality.',
      {'method':'executed','expected':'臺→台:1edit/5=0.2; omitted不:1edit/9≈0.111111111111; empty reference:2edits/0 with cer=None.',
       'observed':cpu['cer_checks'],'tolerance':'Exact integer edit/reference counts; binary float1/9 checked by arithmetic meaning.',
       'details':'Three small CPU probes call actual Unicode DP code; no ASR inference. Ordered-text toy preserves character multiplicities while changing sequence, visually checked in prerequisite figure.',
       'denominators':{'reference_characters_case1':5,'reference_characters_case2':9,'reference_characters_empty_case':0}})
claim('c10','concept','OpenAssistant message trees expand assistant turns with preceding complete path history; all derived turns stay in one family. Course-authored condition examples are text, not new human recordings; repetition does not add independent contexts.','DATA.md lines70-76',
      [evidence('oasst','Dataset Structure; conversation tree example; parent/message/tree IDs; Apache metadata','Original branching messages and source license.'),evidence('chat_helper','main message alternation/history/split assertions','Current exact path expansion and course/source provenance.'),evidence('apache','section4','Source/modified-file/NOTICE retention.'),evidence('manifest','chat rows source/history/family fields','Actual144 chat row declarations and separate provenance.')],['cpu_result','additional'],
      'Source rank/community status is not truth certification, script conversion is not semantic review, and this small supplement does not establish broad Chinese conversational ability. Own code/record audit does not independently judge every adapted chat answer.')
claim('c11','concept','FLEURS and AISHELL recordings are human read speech; selected FLEURS utterances evaluate transcription only, while AISHELL question-shaped utterances use project-declared semantic response criteria on paired gold-text and actual-ASR routes.','DATA.md lines80-88',
      [evidence('fleurs','Dataset Description/Data Fields/Other Known Limitations','Human read speech and source transcripts, not assistant answers.'),evidence('aishell','complete corpus card','Human controlled recordings/manual transcription.'),evidence('aishell_openslr','License/About resource','Apache2.0 speech corpus and controlled reading domain.'),evidence('runtime','run_train lines527-541; evaluation lines861-896/917-935','No audio rows used by SFT; FLEURS speech_transcription bypasses chat; AISHELL uses actual transcript vs source question separately.'),evidence('manifest','audio_rows source transcript/rubric/task fields','38 selected original recordings and null assistant answers, source text preserved.')],['cpu_stdout','cpu_result','fig_natural-v4-asr-two-routes','screen_browser-link-aishell'],
      'No Whisper training, speech-model inference, listening audit or spontaneous conversation benchmark performed here. All38 selected WAV sizes/hashes and sample-frame metadata matched. FLEURS sentence families and source official speaker statement are recorded, individual validation/test speaker identities remain unavailable.')
claim('c12','software','Partial archive reads are labeled as partial acquisition; selected WAV bytes are checked, original Simplified transcripts remain, and manual UI correction must not replace actual-ASR test input with gold text.','DATA.md lines84-88',
      [evidence('voice_helper','BoundedReader/check_audio/replay_archive','Selected members checked against exact bytes/frames, unread archive tail not verified.'),evidence('runtime','evaluation actual=dict(row,user=observation[transcript]) and separate typed route','Actual ASR hypothesis is kept distinct from reference source question.'),evidence('manifest','audio_rows source_archive full_archive_downloaded_and_verified fields and original_spaced_source_transcription','Explicit source-prefix integrity scope and AISHELL whitespace-only input preparation.')],['cpu_stdout','cpu_result','fig_natural-v4-asr-two-routes','view_notes'],
      'This audits stored provenance and source behavior, not full upstream archive identity or a new complete UI/audio evaluation. NFKC/whitespace-normalized CER retains punctuation and script; raw metric remains separate.',
      {'method':'executed','expected':'Actual source loop passes recognized question and original source question as distinct generation inputs, retains the same surrounding conversation/model/options/image/data-root, and skips FLEURS transcription-only chat; selected38WAV hashes/metadata match; compressed voice cap rejects over-budget reads.',
       'observed':{'paired_routes':cpu['paired_route_CPU_probe'],'selected_audio':{'count':len(cpu['audio_metadata']),'metadata_verified':True},'bounded_voice':cpu['behavior_checks']},
       'details':'Compiled and executed the actual current paired-route AST loop at lines917-936 with a generate stub and two small fixtures. Stub calls establish routing/retained-context behavior only, without ASR/LM quality or model execution. Existing selected local WAV hashes/sample frames and provenance records were independently audited; upstream whole archives were not re-fetched. Partial-acquisition labels and original Simplified transcript preservation were read in the manifest and replay source.'})
claim('c13','numeric','Three snapshot download/unpacked sizes and file counts, decimal MB convention and separately rounded total agree with raw manifest bytes.','DATA.md lines116-124',
      [evidence('manifest','archives[].bytes/files; files[].bytes','Original fixed byte and member declarations.')],['cpu_stdout','cpu_result','heads'],
      'Declared file bytes include notices/attribution, exclude filesystem overhead; snapshots/temporary archives can coexist and model/environment storage is additional. HEAD confirms availability/length, not archive body hash.',
      {'method':'executed','expected':{'vision':[69308918,72155176,511],'ocr':[59746570,62382076,961],'voice':[17487821,26146571,41],'totals':[146543309,160683823]},
       'observed':cpu['sizes'],'tolerance':'Exact byte/member arithmetic; display round(bytes/1,000,000,2).',
       'details':'Raw compressed total146543309bytes→146.54MB, unpacked total160683823→160.68MB. Individually rounded columns may differ by0.01MB from rounded total.',
       'denominators':{'bytes_per_decimal_MB':1000000,'snapshot_archives':3,'declared_archive_members':1513}})
claim('c14','software','The three fetch modes list metadata, download/verify fixed snapshot contents, and verify existing unpacked files; conflict contents are retained and rejected. LFS pointer text is not snapshot bytes.','DATA.md lines92-113,126-130',
      [evidence('fetch','main/load_manifest/download_checked/extract_checked/verify_directory/fetch_data','Pin check order, no authorization headers, exact archives/files verification, list/verify mode separation and no-overwrite destination behavior.'),evidence('lfs_spec','The Pointer required keys/example','Small pointer text vs full object.'),evidence('runtime','load_manifest/asset_path lines52-115','Manifest path and explicit data-root are independent.')],['cpu_stdout','cpu_result','additional','final_fetch','heads'],
      'Downloaded only original texts/metadata/example photo for review; never invoked real dataset-download CLI. Existing selected local assets and mocked tiny archive/conflict fixtures establish bounded behavior; three anonymous HEAD requests200 support public availability at retrieval time, not future server reliability.',
      {'method':'executed','expected':'Identical local files accepted; changed bytes/extra file/pointer/path traversal rejected and existing bytes unchanged; loader matches2371rows/38audio/1506physical assets.',
       'observed':cpu['behavior_checks'],'tolerance':'Exact bytes and controlled exception conditions.',
       'details':'Actual imported implementation on a3-byte file and tiny generated tar, plus current manifest/local data and source-parsed flags. Product fetch/install/Git/train shell commands were not run.'})
claim('c15','software','Reconstruction binds fixed9a61ecf code/source selections and exact manifest verification, uses separate CPU environment with the four explicit package versions, writes selected assets to separate specified paths and limits source reads.','DATA.md lines134-165',
      [evidence('build','reconstruct/assemble/validate_assets/package_selected/main','No question/label reselection, no model loading, exact source/manifest/archives comparison.'),evidence('ocr_helper','RangeFile/rebuild encoding/crop/hash guards','Exact206 range or fail,128MiB bounded NVIDIA reads, fixed encoding package versions.'),evidence('voice_helper','SOURCE_LIMITS/replay_archive','Bounded compressed prefixes32/48MiB per source including retries.'),evidence('git_clone','--no-checkout','Separate fixed checkout.'),evidence('lfs_config','GIT_LFS_SKIP_SMUDGE','Skip LFS materialization during checkout.'),evidence('venv','Introduction/creation command','Separate CPU Python environment.'),*[evidence('pkg_'+k,'info.version/summary/requires_python; urls.filename','Pinned package purpose and Python3.13-compatible published wheel.') for k in ['pillow','h5py','numpy','soundfile']]],
      ['cpu_stdout','cpu_result','additional','final_fetch','pinned_probe','format_diff'],
      'Full online reconstruction was not executed; h5py is absent, current SoundFile0.14.0 differs from rebuild pin0.13.1, and no install was authorized for this review. Source AST-equivalent exact pinned OCR separately probed; no bytes-equality claim between its formatted current/pinned source. Byte verification rejects different encoding or server behavior, and future source availability/speed is not guaranteed.',
      {'method':'executed','expected':'Exact fixed source contract, listed CLI flags, published four package versions, Range200/over-cap rejected and cached exact206 accepted.',
       'observed':{'pinned_ocr':cpu['pinned_ocr'],'bounded_behavior':cpu['behavior_checks'],'official_package_metadata':'additional-audit.json official_package_metadata; CLI_help fields'},
       'tolerance':'Exact source AST equality excluding location metadata; byte pins remain separately recorded.',
       'details':'No clone, install, full reconstruction, corpus container download, training or GPU execution. Offline probes exercised Range/cache/limit and voice prefix cap.'})
claim('c16','software','The actual preview displays current DATA and routes its cat/order/source links to the intended current content.','DATA.md linked prerequisites/source navigation',
      [evidence('manifest','source fields and fixed hashes','Linked source files correspond to current selected data.'),evidence('browser','browser.stdout.txt action=goto/click events; browser-receipt.json and visible-text/screenshot files','Actual browser/source provenance evidence, not inferred from HTTP alone.')],['browser_code','browser_execution','browser_receipt','browser_text','screen_browser-DATA.top','screen_browser-link-cat','screen_browser-link-order','screen_browser-link-aishell','final_fetch','view_notes'],
      'Personally read the Chromium-rendered page and clicked20.4 cat,20.11 order and AISHELL source. DATA literal307c325 source SHA equals current complete bytes. Preview predates later10.8/11.2 edits, which were not visited or represented as current HTML; timing metadata remains unfinalized and this is not publication approval.',
      {'method':'executed','expected':'DATA visible question count2,077, total146.54MB and exact9a61ecf rebuild pin; clicked cat→20.4.html, order→20.11.html, AISHELL source→GitHub literal307c325 voice-question-sources.json.',
       'observed':browser_events,
       'details':'Executed existing Chromium through Playwright; personally read captured DATA and clicked-target text, and personally viewed all four screenshots. Also retrieved literal307c325 DATA raw source and compared exact SHA/current fullfile bytes. Browser receipt navigation inventory is incidental, not an assertion that every listed link was clicked or reviewed. No claim about later10.8/11.2 source edits in this older preview.'})

figure_receipts=json.loads((P/'figure-render-receipt.json').read_bytes())
figure_hashes={r['source']:r['source_sha256'] for r in figure_receipts}
for path,d in figure_hashes.items():assert sha(path)==d
report={'schema_version':1,'review_stage':'technical','review_scope_type':'supplemental_whole_document',
        'source':DOC,'reviewer_task':'/root/v4_review_coordinator/factual_guide_data','reviewer_context':'fresh',
        'assigned_document_scope':{'document':DOC,'line_start':1,'line_end':169,'complete_read_to_EOF':True,'bytes':18055,
                                   'scope_note':'Unique assigned whole DATA guide; necessary current prerequisites are explicitly separated below.'},
        'complete_documents':[DOC],'current_document_sha256':{DOC:sha(DOC)},'source_sha256':sha(DOC),'figure_sha256':figure_hashes,
        'completed_on':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_round':2,'verdict':'pass','claims':claims,'sources':sources,'artifacts':artifacts,'issues':[],
        'revision_history':[
            {'round':1,'report_path':str(P/'report.first-issued.raw.json'),'report_sha256':sha(P/'report.first-issued.raw.json'),
             'details':'Actual first-issued own report retained byte-for-byte. Two software claims lacked explicit verification objects and browser evidence was mislabeled as CPU; these reporting gaps were corrected in round2, with an additional meaningful paired-route source probe. No substantive DATA issue or product edit was made.',
             'changed_artifact_preservation':{'directory':str(P/'first-issued-evidence'),'receipt':str(P/'first-issued-evidence/byte-preservation-receipt.json'),
                                               'details':'Round1 CPU code/stdout/result were recovered by removing only the subsequent route-probe addition and verified byte-for-byte equal to their original report SHAs. Original SHA references were never rebound. Other round1 artifacts remain at their recorded bytes.'}},
            {'round':2,'details':'Same-owner final report; explicit actual CPU/browser execution method, expected/observed/details fields for c12/c16, separate browser execution source/artifact, current hashes and truthful stub/model boundaries.'}],
        'checks':{
            'factual_accuracy':{'status':'pass','details':'All16 coalesced substantive claims checked against actual original authorities, current code and scoped record audit; no unresolved or contradicted claim. Annotation correctness outside personally inspected example is not overclaimed.','claim_ids':[c['id'] for c in claims]},
            'numeric_verification':{'status':'pass','details':'All question/material/family/recording units, three archive/member/MB totals and independent recorded eligibility arithmetic recomputed on CPU; current1513local files and38WAV frames checked; empty-CER boundary exercised.','claim_ids':['c02','c04','c09','c13']},
            'figure_consistency':{'status':'pass','details':'DATA itself embeds no SVG. Five actually necessary linked-prerequisite SVGs personally rendered/viewed: cat count/pose, family grouping, artificial relation distinction, reading-order arrow, two ASR/chat routes. No fabricated capability evidence from schematic figures.','claim_ids':['c03','c06','c07','c09','c11','c12']},
            'source_verification':{'status':'pass','details':'Actual original versioned publisher cards, arXiv originals, official licenses,70Commons source entries, Python/Git/LFS/PyPI/software authorities retrieved and read at recorded locators; complete third-party responses stay ignored. Fixed source/manifest retrievals and byte SHA receipts retained. DOCCI PDF bound failure is preserved, successful original v1HTML used.','claim_ids':[c['id'] for c in claims]},
            'limitations':{'status':'pass','details':'Guide distinguishes independent-source vs derivative denominators, course splitting vs unknown foundation pretraining overlap, AI Chinese adaptation vs official human English source, synthetic vs real OCR, transcription vs chatbot answers, read-speech vs spontaneous speech, partial vs complete archive hashes, exact bytes vs content truth, declared storage vs filesystem/model overhead. This owner did no GPU/model/full-container benchmark or full online reconstruction.','claim_ids':[c['id'] for c in claims]}},
        'applicability':{'substantive_claims':True,'reason':'Complete guide contains dataset concepts/counts, licensing, commands and concrete source/reconstruction software contracts; whole-document NA is inappropriate.'},
        'source_read_order':[
            {'path':'outputs/natural-v4/review-plan/supplemental-factual-contract.md','scope':'complete'},
            {'path':'docs/technical-review-guide.md','scope':'complete'},
            {'path':DOC,'scope':'all169lines, raw UTF8 from title through EOF'},
            {'path':'scripts/fetch_natural_data.py','scope':'complete357lines'},
            {'path':'scripts/build_natural_v4_assets.py','scope':'complete493lines'},
            {'path':'course/chapters/20.md','scope':'current20.4/20.5/20.8/20.9/20.10/20.11/20.12 in full'},
            {'path':'docs/natural-assistant/v4/STUDENT.md','scope':'necessary acquisition/environment sections1-2; also read start of section3 through line113'},
            {'path':'scripts/prepare_natural_v4_ocr.py','scope':'complete current313lines; exact pinned source range/rebuild contract and full AST comparison'},
            {'path':'scripts/replay_natural_v4_voice.py','scope':'fixed-URL/row validation, bounded reader/WAV check and full replay/member/retry/CLI logic'},
            {'path':'docs/natural-assistant/v4/manifest.json','scope':'parse complete11,355,009bytes; substantive field projection, source/file/row counts and hash audit; embedded previous data-review judgments are not adopted'},
            {'scope':'Original-authority cards/papers/manuals/legal codes, read versions/locators in sources; actual full response SHA retrieval records. Source order retained in authority receipt arrays.'},
            {'path':'scripts/prepare_natural_v4_chat.py','scope':'complete181lines'},
            {'path':'tiny_perceptron/natural_assistant.py','scope':'necessary loader, LoRA construction, row selection, ASR metrics and two-route evaluation ranges recorded above'},
            {'path':'tiny_perceptron/natural_concepts.py','scope':'actual Unicode DP/CER functions47-67'},
            {'path':'docs/natural-assistant/v4/selection.json','scope':'complete JSON'},
            {'path':'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json','scope':'complete original empirical counts/gate record, own recomputation only'},
            {'path':'docs/natural-assistant/v4/public-release.json','scope':'complete configuration JSON'},
            {'path':'docs/natural-assistant/evidence/v4-research/vision/sources/docci-descriptions.jsonl','scope':'complete source parsed for selected full-caption comparison; train_01827 full original English caption personally read'}],
        'prerequisite_document_sha256':{path:sha(path) for path in ['course/chapters/20.md','docs/natural-assistant/v4/STUDENT.md']},
        'independence_statement':'Fresh fork-none actual reviewer task, unique DATA factual owner distinct from reader/author. No previous reader/factual reports or author-review conclusion files opened, no author identity guessed, no lesson_id invented; historical data-source records were audited only as records.',
        'execution_limits':{'actual_environment':{'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','h5py':'absent','soundfile':'0.14.0'},
                            'performed':'Bounded offline code/number/hash/audio-header probes, existing selected-cache reads, text/metadata retrieval, one64KBthumbnail, archive HEAD only, existing Chromium browsing, Inkscape SVG render and own view_image inspection.',
                            'not_executed':['real data-download CLI','Git clone/checkout','package installation/new environment','full source reconstruction','full upstream compressed containers','model/weights download','GPU or long training','fresh ASR/chat inference','whole book benchmark'],
                            'record_audit_boundary':'Original project GPU/annotation/action history remains recorded history, not this reviewer replication or personal regrading.',
                            'publication_boundary':'Current unchanged DATA source verified against307c325 preview; reading-time metadata and final site publication not adjudicated.'}}
source_ids={s['id'] for s in sources};artifact_ids={a['id'] for a in artifacts}
assert len(source_ids)==len(sources) and len(artifact_ids)==len(artifacts)
for c in claims:
    assert all(e['source_id'] in source_ids and e['locator'] and e['supports'] for e in c['evidence'])
    assert set(c['artifact_ids'])<=artifact_ids
    if c['kind'] in ['software','numeric','empirical']:
        assert all(field in c['verification'] and c['verification'][field] for field in ['method','expected','observed','details'])
    if c['kind'] in ['software','empirical']:
        assert c['verification']['method']=='executed'
        assert any(a['kind']=='execution' and a['id'] in c['artifact_ids'] for a in artifacts)
    if c['kind']=='numeric':assert c['verification']['tolerance']
    if c['kind']=='empirical':assert isinstance(c['verification']['denominators'],dict) and c['verification']['denominators']
for a in artifacts:assert sha(a['path'])==a['sha256']
encoded=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
(P/'report.json').write_bytes(encoded)
if not (P/'report.first-issued.raw.json').exists():(P/'report.first-issued.raw.json').write_bytes(encoded)
print(json.dumps({'report':str(P/'report.json'),'verdict':report['verdict'],'claims':len(claims),'sources':len(sources),'artifacts':len(artifacts),'current_document_sha256':report['current_document_sha256'],'figure_sha256':figure_hashes},indent=2))

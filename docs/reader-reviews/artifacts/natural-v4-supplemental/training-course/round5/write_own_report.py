from pathlib import Path
import json, hashlib, re, copy

p=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/training-course');q=p/'round5'
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda f:json.loads(Path(f).read_text())
save=lambda f,x:Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
prior=read(q/'prior-round4-preserved/report.json');delta=read(q/'current-source-and-exact-delta-receipt.json')
browser=read(q/'browser-receipt.json');figs=read(q/'figure-renewed-view-receipt.json');prereqs=read(q/'prerequisite-byte-comparison.json')
current=Path('course/training.md').read_bytes();old=Path(q/'prior-round4-preserved/training.f9ae45fb62731b514aa704d6cb3677e9b052c6243a2a5c331d9339acbe23bf55.md').read_bytes()
assert H(current)==delta['current_document_sha256'] and old.replace(delta['old_bytes_literal'].encode(),delta['new_bytes_literal'].encode())==current
ls=current.splitlines(keepends=True);context=b''.join(ls[689:814]);(q/'T8-and-needed-context.current-lines690-814.md').write_bytes(context)
code=lambda x:re.findall(rb'^```[^\n]*\n.*?^```[^\n]*$',x,re.M|re.S)
assert code(current)==code(old) and len(code(current))==55
save(q/'source-code-byte-comparison.json',{'source': 'course/training.md','old_full_raw_sha256':H(old),'current_full_raw_sha256':H(current),'all55_blocks_exact_bytes':True,'old_new_block_sha256':[{'ordinal':i,'sha256':H(x)} for i,x in enumerate(code(current),1)],'no_implementation_source_inspected':True,'new_numerical_executions':[]})
source_receipt={'reviewer_task':prior['reviewer_task'],'current_document_sha256':H(current),'current938line_new_whole_source_read_claimed':False,
 'whole_document_assessment_basis':'genuine own round4 full936line read in all8 actual ordered chunks retained; every old/current byte compared and exact single heading replacement proved; complete affected T8 and context newly read in current form, not just hash substitution',
 'retained_true_round4_complete_read_receipt':str(p/'round4/own-complete-source-read-receipt.json'),
 'retained_true_round4_document_snapshot':str(q/'prior-round4-preserved/training.f9ae45fb62731b514aa704d6cb3677e9b052c6243a2a5c331d9339acbe23bf55.md'),
 'new_actual_personal_read_order':[{'ordinal':1,'source':'course/training.md','current_scope_lines':[690,759],'actual_complete_read':True,'method':'untruncated source tool text; current wholeT8 first part plus needed adjoining T7 context'},
 {'ordinal':2,'source':'course/training.md','current_scope_lines':[760,814],'actual_complete_read':True,'method':'untruncated source tool text; current wholeT8 remainder/newheading/alllimits/commands/exercise plus T9 transition'},
 {'ordinal':3,'source':'course/chapters/16.md','section':'16.1','current_scope_lines':[5,47],'actual_complete_read':True,'method':'complete current explicitly linked prerequisite, source snapshot and untruncated output'},
 {'ordinal':4,'source':'course/chapters/15.md','section':'15.11','current_scope_lines':[344,395],'actual_complete_read':True,'method':'complete current 16.1 explicitly linked timing prerequisite; no further hidden explanations needed'}],
 'current_complete_T8_scope_lines':[701,811],'needed_context_snapshot':str(q/'T8-and-needed-context.current-lines690-814.md'),'needed_context_raw_sha256':H(context),
 'no_other_reviewer_or_author_expected_verdict_consulted':True}
save(q/'own-source-read-scope-receipt.json',source_receipt)
screens=[x['screenshot'] for x in browser['training']['T8_screens']]+[s['screenshot'] for x in browser['prerequisites'] for s in x['screens']]+[x['screenshot'] for x in browser['actual_links'] if 'screenshot' in x]+[str(q/'16.1-CPU-output-right-columns.png')]
visible={'current_training_HTTP_sha256':browser['training']['HTTP']['HTTP_raw_sha256'],'current_full_article_visible_sha256':browser['training']['full_visible_text_sha256'],
 'whole_current_browser_new_fullread_claimed':False,'entire_article_byte_reuse_proof':str(q/'complete-visible-byte-reuse-receipt.json'),
 'new_actual_complete_visible_read':['T8 complete article text 3660-token untruncated read','16.1 complete actual article including profiler CPU output','15.11 complete actual article including Dense/MoE CPU output'],
 'new_actual_personally_viewed_screens':screens,
 'all11TOC_actual_clicks':True,'actual_new_measurement_TOC_click':True,'actual16_1_to_new_H3_click':True,
 'actual3guide_scope':'navigation click, h1, full opening only; all opening bytes identical prior own round4; no full guide read claim',
 'actual_guide_openings_personally_read':[x['visible_label'] for x in browser['guides']],
 'guide_purpose':{'操作指引':'按打字→圖片→聲音打開Qwen3-VL/Whisper已有成品；不用先做本頁全套訓練。','資料與來源':'照片/中文轉寫/聊天對上用途並以家族切分；錄音驗收兩站、不重訓Whisper；候選LoRA需驗證。','自行重訓':'另有GPU環境建立圖文語言LoRA候選並與原底座驗證；建立與採用分開，不重訓ASR。'},
 'actual_prev_next_purpose':{'上一頁 名詞對照':'返回術語與例子入口。','下一頁 本章導讀':'進入第一章從下一字開始的導讀。'},
 'root_CPU_provenance':'actually personally read/viewed rootbuild16.1/15.11 outputwidgets; no own numerical rerun; training itself zero outputwidgets',
 'officialAPI_reuse':'T8 official2.14 href unchanged under complete single-source-change proof; retain own round4 actual destination/complete API read/image evidence; no round5 fresh external retrieval claimed'}
save(q/'own-visible-read-receipt.json',visible)
sourceversion={'reported_literal_revision':'160fd47dede5c2453c8a08dd535290ddfd0b915d','coordinator_provided_literal':True,'no_Git_inspected':True,
 'actual_browser_localbuild':browser['training']['HTTP'],'actual_visible_DOM_revision_pinned_hrefs':browser['training']['actual_revision_pinned_hrefs'],'current_document_sha256':H(current)}
save(q/'source-version-provenance.json',sourceversion)
sections=copy.deepcopy(prior['source_read_sections'])
for x in sections:
 x['read_basis']='retained genuine own round4 complete936line read; exact allremaining current bytes unchanged after the one measured heading transformation; not new round5 whole-source read'
 if x['section']=='T.8':
  x['lines']='701–811';x['read_basis']='new complete current T8+neededcontext source690–814 and actual browser completeT8 text/all3tiles, read personally in round5'
  x['own_summary']='同資料/種子/更新數先保存基準，只換一項；modern六組240與MoE七組180各有自己的目標數和資源代理。效率另外用直接SFT底座45/5/10家族，四頭MHA/GQA各100、四主MHA副本各40；padded/packed只學助手短文，品質與三記憶體欄不同。新選讀H3把L4/FP32/TF32關閉、tensor配置峰值/reset、排除driver/reserved/獨立Inductor程序及MiB邊界集中；16.1現在直接指到這些条件。Flash只是固定QKV核心而非模型訓練；precision從原單頭SFT另起、200嘗試/成功/skip各記。品質、大小、速度各需量測。'
 if x['section'] in ('T.9','T.10','T.11'):
  a,b=map(int,x['lines'].split('–'));x['lines']=f'{a+2}–{b+2}'
issue=copy.deepcopy(prior['own_issues'][0]);issue.pop('round4_own_adjudication',None)
issue['location']['page']='http://127.0.0.1:8789/training.html'
issue['location']['evidence']=[str(q/'browser-receipt.json'),str(q/'measurement-subheading-TOC.png')]
issue['round5_own_adjudication']='本人当前11次TOC点击与newH3截图仍见h2/TOC只有题名；新增子标题帮助方法定位但未显示T编号。保留open/nonblocking，由正确anchors仍可完整阅读；不因产品选择关闭、不从其他owner结果推PASS。'
checks=[
 {'name':'background_and_links','verdict':'pass','own_assessment':'保留本人真936行全文理解，逐完整rawbyte证明当前只一处heading改动后，完整新读T8及16.1/真正需用的15.11。16.1的量测方法标签真点击直接到新H3，其内容是配方/起点/记忆体/计时边界；T8架构入口仍独立有用。所有11Tanchors和三guide/前后实际导览用途可理解。'},
 {'name':'terminology_and_symbols','verdict':'pass','own_assessment':'当前T8完整新读后，参数格数、容量代理、FLOPs、有效目标、NLL/答对/EOS、tensorallocated峰值与reserved/driver仍可分清。16.1使Self/total/avg/calls含义明晰：total不能父子累加，CPU活动不量GPU；MiB2²⁰、9次中位数与30次平均属不同量测。新H3未改这些词或数字。'},
 {'name':'examples_and_figures','verdict':'pass','own_assessment':'T8表与完整3tiles重新亲看，6modern/7MoE/8效率更新组、Flash两精度核心和3AMP支线的材料及预算不能拼成一条权重路线。16.1/15.11实际CPU例子显示形状/bytes与时间分工，并读了可见右栏。9必要SVG及本人originalChromiumPNG exactbyte比对后再次view_image；未声称新render。其余13表/例子沿用原真实全文理解及完整visiblebyte比对。'},
 {'name':'programs_and_commands','verdict':'pass','own_assessment':'所有55current代码/命令与ownround4逐byte相同，T8本轮完整逐项读。prepare_data产生三侧后train只用指定train；evaluate输出完整样本/有效分母；modern/moe从故事包，efficiency/precision需SFT权重/同目录资料，flash不需要底座/数据但需要CUDA。16.1/15.11是CPU前向定位/计时、不训练，真正输出是rootbuild已跑结果。本轮没有数值执行。'},
 {'name':'exercises_and_workflow','verdict':'pass','own_assessment':'本轮新读的T8练习先列假设、固定资料/种子/步数并同时留质量成本；16.1改12token仍应24416权重bytes，用同名工作多轮比较、无稳定差距也可完成；15.11改长度64重测比值且不预设胜者。原T1–T11练习/失败界限无变化，保持此前本人实际理解。新方法跳转减少定位步骤，原T编号可见性问题仍open但不阻碍。'}]
commands=[{'group':'small architecture exercise','inputs':'seed42 generatedtoy-text train; unchanged three200-step data/seed settings','outputs':'baseline/rms/moe weights and JSON parameters/seconds; separate samevalidation evaluation JSON with mean_token_nll/effective_tokens/samples/skipped','limit':'one heldout row only; samewidth not equalparameters/FLOPs; source commands read, not executed'},
 {'group':'modern and MoE formal routes','inputs':'fixed512whole stories split409/51/52; modern240updates452102targets versus MoE180updates337761targets','outputs':'sixmodern/sevenMoE named checkpoints+3datasets; model.pt fixedcopybaseline or top2aux.01, not selected bytest','limit':'no seed/largeGPU/generalquality replication; generated eight examples not allheldout'},
 {'group':'efficiency','inputs':'T4 directSFTmodel.pt+dataset.json family45/5/10; original5/10; fourQuery-head branchMHA/GQA100updates then fourMHAcopies40','outputs':'sixprimarytrainingpath allocatorstart/peak/delta; padded/packed40 each have changedtargets/no same3memoryfields; model.pt ordinary fixedcopy','limit':'tensorGPUspace in thisprogram only; no driver/unusedreserved/independentInductor; readnotrun'},
 {'group':'Flash probe','inputs':'fixedQ/K/V[2,4,512,32] FP16/BF16 compatibleCUDA, noSFT/data dependency','outputs':'fixture notmodelweights; result.json kernels/profiler/tolerance/gradfinite/timing/peaks/env','limit':'independent coreprobe withoutloss/optimizer/modelquality; no run'},
 {'group':'precision','inputs':'same originalsingle-head SFTweights+dataset anew, not continuationfromfour-head efficiency','outputs':'FP32/BF16/FP16 each200attempts success200skip0; weight/Adam FP32; fixedmodel.pt FP32','limit':'AMPname does notguarantee speed/memory/quality; readnotrun'},
 {'group':'16.1 profiler and15.11 timing examples','inputs':'TinyLMwidth8 IDs[1,2,3], CPUone thread3warm5profile; Dense/MoE fixedseed[2,8,8],5warm30timed','outputs':'rootactual(1,3,264),24416bytes,self/total/avg/calls; Dense280/.033ms andMoE336/.4281ms','limit':'actualrootbuild outputs read/viewed, not own run; profiler combinedsameoperation notuniquelayer; timingnot universal'}]
extra=[read(q/'prerequisite-16.1-source-receipt.json'),read(q/'prerequisite-15.11-source-receipt.json')]
report={'reviewer_task':prior['reviewer_task'],'round':5,'role':prior['role'],
 'fresh_context':'Original same canonical fullcourse first-reader owner corrective changed-scope round; original first fresh identity and all genuine prior reports retained. No newly invented reviewer identity or new whole938line read claimed. No other review/authorreason/expectedverdict/Git/implementation/checker consulted.',
 'contract':prior['contract'],'actual_assignment':'current wholecourse judgment based on genuine round4 complete936line source read, exact fullbyte localization of oneheading change, new complete currentT8+neededcontext and complete current16.1/needed15.11; actual8789browser newheading/directlink/TOC/prerequisites/needed navigation; nineSVG exactreuse plus renewedview',
 'current_document_sha256':H(current),'reported_literal_source_revision':sourceversion['reported_literal_revision'],'source_version_provenance':str(q/'source-version-provenance.json'),
 'documents':[{'repository_relative_file':'course/training.md','full_file_raw_utf8_sha256':H(current),'raw_bytes':len(current),'actual_line_count':len(ls),'preserved_snapshot':delta['current_snapshot'],
  'current_round5_complete_whole938line_new_read':False,'own_previous_round4_complete936line_read_retained':True,'complete_document_coverage_basis':source_receipt['whole_document_assessment_basis'],
  'actual_read_order':{'previous_complete_order':prior['documents'][0]['actual_read_order'],'previous_document_sha256':H(old),'new_current_changed_scope_order':source_receipt['new_actual_personal_read_order']},
  'own_summary':'從有限任務、家族留出與可保存比較起點開始，區分通路檢查、實際更新、獨立模型路線和同一權重接續；品質/格式/EOS/成本/容量預算各需自己的數字。新效率量測H3把已有条件集中，16.1可直接到正方法。它不增加实验结论，帮助读者追到起点、范围与重跑输入。',
  'understanding_assessment':'沿用本人真实完整旧文本理解，完整currentbyteproof定位改动，并以本轮完整T8/必要前置/actualbrowser语义重新判断当前整份训练指南。可解释command输入/输出、模型/资料/计数角色与局限，判可理解；非事实或全书阶段闭合。'}],
 'source_read_sections':sections,'necessary_prerequisites':{'retained_29_complete_own_sections_current_exact_bytes':prereqs,'new_round5_complete_necessary_source_sections':extra,
  'own_needed_context_reason':'T8显式引16.1时间同步/定位原理；16.1显式引15.11计数与真时间的差别。本轮完整读这两段。4.5已经本人完整读、当前逐byte相同；15.13及其余29段本人旧完整阅读经exactbyte证据沿用。15.11本身已解释分派/计数/暖机，原15.13理解足以接上，没有为这次方法定位继续打开不必要章节。',
  'own_prerequisite_summaries':{'16.1':'用宽8完整模型CPU前向profiler定位运算类型；3暖5记录，形状与24416权重bytes核通路。Self与total区别、calls多于5有层重复；同名汇总非唯一层，权重大小非工作峰值。GPU同步9中位数和训练更新边界分开。','15.11':'少用矩阵权重不一定快，索引/分派/合并也耗时；固定CPUseed线程输入，5暖30平均。Dense280/MoE336整体参数区别于矩阵粗估256/160；不训练不比质量，小CPU耗时不代表大型GPU。L4表每支线reset但起点不同，需看增量。'},'no_whole_otherchapter_or3guide_read_claim':True},
 'original_svg_figure_map':figs,'pages':{'current_training':browser['training'],'current_complete_needed_prerequisite_articles':browser['prerequisites'],'current3guide_landings':browser['guides'],'own_prior_unchanged_browser_evidence':str(p/'round4/own-visible-read-receipt.json'),'current_own_visible_read_receipt':str(q/'own-visible-read-receipt.json')},
 'actual_navigation':{'current_actual_links':browser['actual_links'],'current11TOC_clicks':browser['toc_clicks'],'current_prev_next':browser['navigation'],'current3guides':browser['guides']},
 'own_workflow_data_and_models':prior['own_workflow_data_and_models'],'workflow_explanation_reuse_basis':'previous own complete actual source understanding retained because all content/code/numbers except oneheading exactsame; T8current summary and commandroles freshly explained above',
 'own_current_commands_count_roles_limits':commands,'five_checks':checks,
 'own_exercise_predictions_and_checks':[{'section':'T8','source_guided_expectation':'control one condition and keep split/seed/updates; quality/cost reported independently','actual_check':'new fullsource/browser read, comparison budget/role/limits explained; no experiment rerun'},
 {'section':'16.1','source_guided_expectation':'12token→(1,12,264),same24416weightbytes; no fixedmicrosecond outcome','actual_check':'paper shape/bytes check and rootoriginal3token result personally read/viewed;12token exercise notexecuted, no preregistration claim'},
 {'section':'15.11','source_guided_expectation':'Dense256+16+8=280; MoE4*(64+4+8)+32=336, elapsednot universallyfixed','actual_check':'paper calculation and actualrootoutput280/336 read/viewed; length64 exercise unexecuted'},
 {'section':'allunchangedT1–T11','actual_check':'retained own original exercises/commands evidence, exactcontentreuse proof; no newfullguideexercise/CPUexecution claim','prior_own_receipt':str(p/'round4/report.json')}],
 'execution_evidence':{'own_round5_numerical_executions':[],'own_original_only_CPU_reuse':str(q/'original-own-CPU-reuse-receipt.json'),'current_actual_rootCPU_widgets':'16.1/15.11 personally read/viewed existing output, not my numerical execution','actual_browser_commands':[str(q/'browser_round5.py'),str(q/'browser_output_scroll.py')],'actual_browser_outputs':[str(q/'browser-run-output.txt'),str(q/'browser-output-scroll.stdout.txt')],'SVG_original_render_command_output_receipts':'original own round1 receipts preserved in original artifact tree; exact PNG reuse and newactualview receipts round5/figure-renewed-view-receipt.json'},
 'verdict':'PASS','verdict_reason':'本人保留真全936行閱讀、精確驗證唯一currentheading改動後，完整重讀受影響T8和必要16.1/15.11、實際新站語義和圖。新H3的落點正是量測條件，正文范围與命令角色清楚，沒有新阻礙；編號可見性仍非阻礙open。裁決限本人指定整份訓練指南可讀性。',
 'own_issues':[issue],'new_issues':[],'resolved_own_issues':[],'remaining_issues':[issue],
 'corrective_change_readability_assessment':'把原有bold段首分成H3，16.1的量测方法直接到conditions，右TOC同名子項可点，T8架构主入口保留。此更动改善方法定位，未关闭既有T编号可见性issue，也未制造新实验或改变CPU数字。',
 'prior_rounds_preservation':{'round4_report_sha256':'73855c0c981ceae8c02891ce5cd2a81b0f9c4234217a6387b2889f44ab55b122','receipt':str(q/'prior-round4-preservation-receipt.json'),'exact_archive':str(q/'prior-round4-preserved'),'round1_2_3_evidence':'all earlier genuine own reports/source/CPU/browser/render files retained in original artifact history and validated via own earlier manifests; no overwrites'},
 'own_evidence_collection_attempts':[delta['own_initial_collection_attempt'],{'receipt':str(q/'browser-attempt-01-receipt.json'),'kind':'own initialencoded/visibleTOC selector timeout; preserved true attempt before correcting own browser script; real successful actualnav incurrentreceipt; not a product finding'}],
 'scope_limits':['READABILITY ONLY; no FACT/benchmark/implementation/wholebookclosure claim.','Currentguide123957bytes938lines; historical genuine936line complete source reading retained, newscopeT8/context690–814+16.1/15.11 whole subsections; no fabricated currentfull938line reread.','Entire current source and entirearticlevisible bytes compared; unchanged content honestly reused own prior complete readings.','No new numerical CPU/GPU kernels, download/install/train/build/publish/spawn or product/checker/Git/env/time/minutes edits.','Threeguides only actuallanding/opening; no wholeguide read; additionalchapters only explicit genuinely needed subsections.','9originalSVG exactsame andownPNG exactsame; renewedview_image for all9, not rerender; originaltext/sftCPU predictions/results reused as past only.','Official2.14 destination remains sameactualhref under exactwholeguide proof; originalownround4 realAPIview retained, no newofficialpage retrieval claimed.'],
 'actual_evidence_paths':[str(x) for x in sorted(q.iterdir()) if x.is_file()]+[str(q/'figure-renewed-view-receipt.json'),str(q/'own-source-read-scope-receipt.json'),str(q/'own-visible-read-receipt.json')]}
save(q/'report.json',report)
assert H((q/'prior-round4-preserved/canonical.report.json').read_bytes())=='73855c0c981ceae8c02891ce5cd2a81b0f9c4234217a6387b2889f44ab55b122'
(p/'report.json').write_bytes((q/'report.json').read_bytes())
print(json.dumps({'report':str(q/'report.json'),'report_sha256':H((q/'report.json').read_bytes()),'current_document_sha256':H(current),'current_actual_lines':len(ls),'verdict':report['verdict'],'remaining_issues':[{'id':i['id'],'status':i['status'],'severity':i['severity']} for i in report['remaining_issues']]},ensure_ascii=False,indent=2))

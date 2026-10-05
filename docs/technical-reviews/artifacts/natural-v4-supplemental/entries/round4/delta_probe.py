from pathlib import Path
import difflib,hashlib,html,json,os,subprocess,sys,tempfile,textwrap
import pytest,torch
OUT=Path(__file__).resolve().parent;PRIOR=OUT.parent;ROOT=OUT.parents[5];R3=PRIOR/'round3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
r=read(R3/'report.json');expected_report='4db9ccf8364d5e9b425f2dbe9811edbc5c6efd9c3b565c7db87c2b133cc1455e'
assert sha(R3/'report.json')==expected_report
before=ROOT/'docs/natural-assistant/evidence/v4-research/ci-failure-summary-diagnostics/ci.before.yml.source.txt';after=ROOT/'docs/natural-assistant/evidence/v4-research/ci-failure-summary-diagnostics/ci.after.yml.source.txt';current=ROOT/'.github/workflows/ci.yml'
old=r['sources'][next(i for i,s in enumerate(r['sources']) if s['id']=='r_ci')]
assert sha(before)==old['sha256']=='ff06a806d38f66f3805205e9b46003e87b411fd884d04eee54c300da3c6dd03c'
assert after.read_bytes()==current.read_bytes() and sha(current)=='52290167d796430e3c0685e17b8369061a2588175689423f87dfc5c43464aff3'
(OUT/'ci.before.read-snapshot.yml').write_bytes(before.read_bytes());(OUT/'ci.after.read-snapshot.yml').write_bytes(current.read_bytes());(OUT/'prior-round3-report.read-snapshot.json').write_bytes((R3/'report.json').read_bytes())
diff=''.join(difflib.unified_diff(before.read_text().splitlines(keepends=True),current.read_text().splitlines(keepends=True),fromfile='registered prior ci.yml',tofile='actual current ci.yml'))
(OUT/'ci.actual-delta.diff.txt').write_text(diff)
oldtext=before.read_text();newtext=current.read_text();oldstep='      - run: uv run ${{ matrix.extra }} pytest\n';newstep='      - run: uv run ${{ matrix.extra }} pytest --junitxml=outputs/pytest-results.xml\n'
assert oldtext.endswith(oldstep);prefix=oldtext[:-len(oldstep)]
assert newtext.startswith(prefix+newstep)
added=newtext[len(prefix+newstep):]
assert 'if: failure()' in added and 'shell: bash' in added and 'continue-on-error' not in newtext
assert oldtext.split('      - run: uv run ${{ matrix.extra }} pytest')[0]==newtext.split('      - run: uv run ${{ matrix.extra }} pytest')[0]
body=newtext.split("          uv run ${{ matrix.extra }} python - <<'PY'\n",1)[1].rsplit('          PY\n',1)[0]
body=textwrap.dedent(body);compile(body,'actual_workflow_summary.py','exec');(OUT/'actual-workflow-summary.py').write_text(body)
record={'reviewer_task':'/root/v4_review_coordinator/factual_whole_entries','review_round':4,'declared_source_freeze':'d9dc678a57c381c8b57ebf7912e8175d9eef9cc3','environment':{'python':sys.version,'pytest':pytest.__version__,'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available())},'scope':'Own necessary workflow-authority delta review only; existing complete four-document factual verdict/read/browser evidence explicitly reused after exact-byte checks. Diagnostic parser execution uses synthetic XML fixtures and is not a hosted CI run, test-suite rerun, Windows root-cause repair or success claim. No Git/browser/model/data/training/setup action.'}
assert torch.__version__=='2.14.1+cpu' and not torch.cuda.is_available()
record['preserved_prior_reports']=[]
for p,h in [(PRIOR/'report.json','d9482cdfc38766e00160b823235f819090b24d7e512c88103cf6d94a25abb5c5'),(PRIOR/'round2/report.json','8037f2307fda5919c639162d8532b454baf42379349386c2c063a9871ba66310'),(R3/'report.json',expected_report)]:
 assert sha(p)==h;record['preserved_prior_reports'].append({'path':str(p.relative_to(ROOT)),'sha256':h,'verdict':read(p)['verdict']})
record['complete_document_byte_identity']={}
for f,h in r['current_document_sha256'].items():
 assert sha(ROOT/f)==h;record['complete_document_byte_identity'][f]={'current_sha256':h,'last_complete_read_sha256':h,'exact_equal':True,'inspection':'Own actual previous complete reading retained; no unnecessary new fullread claimed.'}
record['figure_byte_identity']={}
for f,h in r['figure_sha256'].items():assert sha(ROOT/f)==h;record['figure_byte_identity'][f]={'current_sha256':h,'last_personally_rendered_sha256':h,'exact_equal':True}
record['registered_code_comparison']=[]
for s in r['sources']:
 if s['kind']=='repository_code':
  actual=sha(ROOT/s['path']);same=actual==s['sha256'];assert same or s['id']=='r_ci'
  record['registered_code_comparison'].append({'source_id':s['id'],'path':s['path'],'previous_sha256':s['sha256'],'current_sha256':actual,'exact_equal':same})
record['original_authority_text_byte_identity']=[]
for s in r['sources']:
 if 'raw_retrieval_path' in s:
  assert sha(ROOT/s['raw_retrieval_path'])==s['retrieval_sha256'];record['original_authority_text_byte_identity'].append({'source_id':s['id'],'sha256':s['retrieval_sha256'],'exact_equal':True})
record['registered_evidence_byte_identity']=[]
for a in r['artifacts']:
 assert sha(ROOT/a['path'])==a['sha256'];record['registered_evidence_byte_identity'].append({'artifact_id':a['id'],'path':a['path'],'sha256':a['sha256'],'exact_equal':True})
prior=read(R3/'current-record.json');record['fixed_original_record_byte_identity']=[]
for a in prior['fixed_records_unchanged']:
 assert sha(ROOT/a['path'])==a['sha256'];record['fixed_original_record_byte_identity'].append({'path':a['path'],'sha256':a['sha256'],'exact_equal':True})
help_result=subprocess.run([sys.executable,'-m','pytest','--help'],cwd=ROOT,capture_output=True,text=True,check=True)
selected=[l for l in help_result.stdout.splitlines() if 'junitxml' in l or 'junit-xml' in l or 'Create junit-xml' in l];assert selected
record['actual_cli_check']={'argv':[sys.executable,'-m','pytest','--help'],'exit_code':help_result.returncode,'pytest_version':pytest.__version__,'observed_help_lines':selected,'scope':'Help only; no test suite or hosted runner execution.'}
fixture='<testsuites><testsuite><testcase classname="unit&lt;unsafe&gt;" name="unicode中文&amp;name"><failure message="fallback&lt;message&gt;" /></testcase><testcase classname="unit" name="error"><error>trace &lt;script&gt; &amp; detail</error></testcase><testcase classname="unit" name="pass"/><testcase classname="unit" name="skip"><skipped/></testcase></testsuite></testsuites>'
record['actual_extracted_summary_execution']=[]
for name,xml in [('missing_report',None),('failure_error_pass_skip',fixture)]:
 with tempfile.TemporaryDirectory(prefix='diagnostic-fixture-',dir=OUT) as temp:
  folder=Path(temp);summary=folder/'summary.md';summary.write_text('PREEXISTING SUMMARY\n');(folder/'outputs').mkdir()
  if xml is not None:(folder/'outputs/pytest-results.xml').write_text(xml)
  env={**os.environ,'GITHUB_STEP_SUMMARY':str(summary)}
  run=subprocess.run([sys.executable,str(OUT/'actual-workflow-summary.py')],cwd=folder,env=env,capture_output=True,text=True)
  assert run.returncode==0 and not run.stderr
  observed=summary.read_text();assert observed.startswith('PREEXISTING SUMMARY\n')
  if xml is None:assert 'No JUnit report was written.' in observed
  else:
   assert observed.count('<details>')==2 and 'unit&lt;unsafe&gt;::unicode中文&amp;name' in observed and 'fallback&lt;message&gt;' in observed and 'trace &lt;script&gt; &amp; detail' in observed
   assert '::pass' not in observed and '::skip' not in observed and '<script>' not in observed
  (OUT/(name+'.summary.md')).write_text(observed)
  record['actual_extracted_summary_execution'].append({'fixture':name,'input':'Absent XML' if xml is None else xml,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'output_file':name+'.summary.md','observed':observed,'limits':'Synthetic fixture executes exactly extracted current Python body on local CPU. Not a real CI failure or Windows run.'})
record['own_delta_assessment']={'changed_source_id':'r_ci','prior_sha256':sha(before),'current_sha256':sha(current),'own_read_scope':'Complete before/after workflow source bytes personally read and independently compared; added pytest argument and entire failure summary Python personally inspected.','retained_matrix':'Linux/Windows/macOS matrix,uv sync,check_env,pytest selection/return behavior and all preceding steps remain byte-identical.','pytest_delta':'Only report-generation argument --junitxml=outputs/pytest-results.xml is added; no continue-on-error or pass override. Actual installed pytest help confirms argument syntax.','summary_delta':'if:failure() adds diagnostic appendix using current XML failure/error nodes; skips pass/skipped nodes,escapes names/messages,appends to GITHUB_STEP_SUMMARY,and states missing XML leaves original failing step/exit authoritative.','affected_claims':['c13'],'historical_result_scope':'Original CI run37071683289/3OS success and existing GPU/MPS denominators remain historical retained records. This change does not repair or newly validate Windows and does not imply current hosted CI success.','independent_judgment':'Current workflow still supports c13 current three-OS configuration claim; original success evidence and all authored document statements remain unchanged. No unresolved factual issue introduced within assigned scope.'}
(OUT/'delta-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Actual authority delta',record['own_delta_assessment'])
print('Actual bounded diagnostic fixtures',[(x['fixture'],x['exit_code']) for x in record['actual_extracted_summary_execution']])
print('Preserved reports',[x['sha256'] for x in record['preserved_prior_reports']])

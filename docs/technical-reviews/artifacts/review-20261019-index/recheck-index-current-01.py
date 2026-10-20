"""唯讀複查首頁與其變動來源，將短算例、差異及沿用證據收據輸出為 JSON。"""
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scripts import export_course
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, log_mel, scene, tone
from tiny_perceptron.selftrained.model import AudioEncoder


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    result=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,check=True)
    return {'command':['git',*args],'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}


def fetch(url):
    command=['curl','--fail','--silent','--show-error','--location','--max-time','40',url]
    result=subprocess.run(command,capture_output=True,check=True)
    return result.stdout


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    original_path='docs/course-review-20261019/reviews/rv1019_technical_index/technical-index-original-01.json'
    original_raw=(ROOT/original_path).read_bytes()
    assert sha(original_raw)=='63d84feba98caa06b55976f38de360a86b934be1c29c2c16240a6a47a6b5681c'
    original=json.loads(original_raw)
    old_run=json.loads((ROOT/'docs/technical-reviews/artifacts/review-20261019-index/audit-output.json').read_bytes())
    inherited=[{'id':a['id'],'path':a['path'],'expected':a['sha256'],'observed':sha((ROOT/a['path']).read_bytes())} for a in original['artifacts']]
    assert all(x['expected']==x['observed'] for x in inherited)
    source_receipts=[{'id':s['id'],'path':s['path'],'previous_sha256':s['sha256'],'current_sha256':sha((ROOT/s['path']).read_bytes())} for s in original['sources'] if s['kind']=='repository_code']
    changed=[x['path'] for x in source_receipts if x['previous_sha256']!=x['current_sha256']]
    assert set(changed)=={'course/chapters/01.md','course/chapters/10.md','course/chapters/12.md'}
    diffs=[]
    for filename,base in [('course/chapters/01.md','f47d47b91432'),('course/chapters/10.md','5ef56bc756f3'),('course/chapters/12.md','5ef56bc756f3')]:
        before=git('show',base+':'+filename)['stdout']
        after=(ROOT/filename).read_text()
        target=git('show','6a7aa1de5227:'+filename)['stdout']
        diff=git('diff',base,'6a7aa1de5227','--',filename)
        blocks=lambda text:re.findall(r'^```python\n(.*?)^```',text,re.M|re.S)
        expected=next(s['sha256'] for s in original['sources'] if s.get('path')==filename)
        assert target==after
        assert sha(before.encode())==expected
        assert blocks(before)==blocks(after)
        diffs.append({'path':filename,'base':base,'base_sha256':sha(before.encode()),'current_sha256':sha(after.encode()),'matches_requested_target':True,'python_blocks_unchanged':True,'python_block_count':len(blocks(after)),'chapter_introduction_unchanged':before.split('\n## ')[0]==after.split('\n## ')[0],**diff})
    frozen_path='docs/course-revision-20261016/freeze-01/sources/index.md'
    frozen=(ROOT/frozen_path).read_bytes()
    freeze_path='docs/course-review-20261019/freeze-06.json'
    freeze_raw=(ROOT/freeze_path).read_bytes()
    freeze=json.loads(freeze_raw)
    page=next(x for x in freeze['pages'] if x['page_id']=='index')
    assert sha(freeze_raw)=='9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9'
    index=json.loads((ROOT/'course/lesson-index.json').read_bytes())
    home_same=export_course.home_introduction(index,True).encode()==frozen
    assert home_same and sha(frozen)==original['source_sha256']==page['source_sha256']
    assert page['figures_sha256']=={}
    headings=[]
    for p in sorted((ROOT/'course/chapters').glob('*.md')):
        headings.extend(re.findall(r'^## ([A-Z\d]+\.\d+) (.+)$',p.read_text(),re.M))
    assert {(x['id'],x['title']) for x in index}==set(headings) and len(index)==len(headings)==312
    footers=[]
    for item in index:
        nb=json.loads((ROOT/item['notebook']).read_bytes())
        footer=export_course.section_footer(item,nb,'main',True)
        footers.append({'id':item['id'],'download':f'href="{item["notebook"]}" download' in footer,'colab':f'/blob/main/{item["notebook"]}' in footer})
    assert all(x['download'] and x['colab'] for x in footers)
    opening=re.findall(r'^```python\n(.*?)^```',(ROOT/'course/chapters/01.md').read_text(),re.M|re.S)[0]
    opening_run=subprocess.run([sys.executable,'-c',opening],capture_output=True,text=True,check=True)
    assert opening_run.stdout==old_run['fresh_opening_example']['stdout']
    gradients=[]
    for coefficient in [0.5,2.0]:
        w=torch.tensor([1.0,2.0],requires_grad=True)
        loss=(w[0]-3).square()+coefficient*w[1].square()
        loss.backward()
        gradients.append({'coefficient':coefficient,'loss':loss.item(),'gradient':w.grad.tolist()})
    assert gradients[-1]['loss']==12 and gradients[-1]['gradient']==[-4.0,8.0]
    image=torch.tensor([[1.0,0.0],[0.0,1.0]])
    text=torch.tensor([[0.0,3.0],[2.0,0.0]])
    similarity=F.normalize(image,dim=-1)@F.normalize(text,dim=-1).T
    try:
        F.normalize(torch.tensor([[0,3],[2,0]]),dim=-1)
    except RuntimeError as error:
        integer_error=str(error)
    else:
        raise AssertionError('整數輸入未按預期拒絕')
    close_a=torch.tensor([1.0+1.5e-5],dtype=torch.float64)
    close_b=torch.tensor([1.0],dtype=torch.float64)
    default_close=bool(torch.allclose(close_a,close_b,atol=1e-5))
    absolute_only=bool(torch.allclose(close_a,close_b,atol=1e-5,rtol=0))
    assert default_close and not absolute_only
    frame=tone()[:400]
    window=torch.hann_window(400)
    energies={'unweighted':float(frame.square().sum()),'hann':float((frame*window).square().sum()),'ones':float((frame*torch.ones(400)).square().sum())}
    assert abs(energies['unweighted']-50)<1e-4 and abs(energies['hann']-18.75)<1e-4
    periodic=[]
    n=torch.arange(401,dtype=torch.float64)
    for frequency in [440,450]:
        wave=0.5*torch.sin(2*math.pi*frequency*n/16000)
        periodic.append({'frequency':frequency,'cycles_in_frame':frequency*400/16000,'sample_at_frame_start':float(wave[0]),'sample_at_last_stored_index':float(wave[399]),'sample_at_next_period_start':float(wave[400]),'dft_non_peak_power':float(torch.fft.rfft(wave[:400]).abs().square().sum()-torch.fft.rfft(wave[:400]).abs().square().max())})
    sequence=log_mel(torch.cat([tone(440),tone(880)])).T
    flipped=sequence.flip(0)
    mean_report={'shape':list(sequence.shape),'same_sequence':bool(torch.equal(sequence,flipped)),'same_log_feature_mean':bool(torch.allclose(sequence.mean(0),flipped.mean(0))),'max_mean_difference':float((sequence.mean(0)-flipped.mean(0)).abs().max())}
    assert not mean_report['same_sequence'] and mean_report['same_log_feature_mean']
    pool_input=torch.arange(44,dtype=torch.float32).reshape(1,1,-1)
    bins=[(i*44//16,math.ceil((i+1)*44/16)) for i in range(16)]
    pooled=F.adaptive_avg_pool1d(pool_input,16)
    manual=torch.stack([pool_input[0,0,start:end].mean() for start,end in bins])
    assert torch.equal(pooled[0,0],manual)
    encoder=AudioEncoder(64).eval()
    encoded=[]
    with torch.no_grad():
        for frames in [173,600,1000]:
            mel=torch.randn(frames,40)
            shortened=encoder.temporal(mel.T[None]).shape[-1]
            tokens,_=encoder(mel)
            assert list(tokens.shape)==[16,64]
            encoded.append({'input_frames':frames,'after_temporal':shortened,'output_shape':list(tokens.shape)})
        language=TinyLM(ModelConfig(width=8)).eval()
        multimodal=MultiModalLM(language).eval()
        tok=ByteTokenizer()
        text_ids=torch.tensor([1,20,30])
        direct=language(text_ids[None])['logits']
        wrapped=multimodal(text_ids)['logits']
        both=multimodal(torch.tensor([1,tok.image_id,tok.audio_id,20,30]),image=scene(),waveform=tone())['logits']
        assert multimodal.language is language and torch.equal(direct,wrapped)
    public=[]
    for label,url,expected in [
        ('home',old_run['live_home']['url'],old_run['live_home']['sha256']),
        ('search_index',old_run['search']['index_url'],old_run['search']['index_sha256']),
        ('search_worker',old_run['search']['worker_url'],old_run['search']['worker_sha256']),
        ('mobile_css',old_run['live_home']['mobile_css_url'],old_run['live_home']['mobile_css_sha256'])]:
        raw=fetch(url)
        public.append({'resource':label,'url':url,'sha256':sha(raw),'previous_sha256':expected,'same_as_original':sha(raw)==expected})
    info=json.loads(fetch('https://birdhackor.github.io/tiny-perceptron-vlm/build-info.json'))
    exact_public_version=info==old_run['live_home']['build_info'] and all(x['same_as_original'] for x in public)
    extra_sources=[{'path':p,'sha256':sha((ROOT/p).read_bytes())} for p in ['tiny_perceptron/selftrained/model.py','tiny_perceptron/natural_concepts.py','tiny_perceptron/model.py']]
    result={
        'command':'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-index/recheck-index-current-01.py',
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'os':platform.platform(),'device':'CPU','torch_threads':str(torch.get_num_threads())},
        'accessed_at_utc':datetime.now(timezone.utc).isoformat(),
        'original_report':{'path':original_path,'sha256':sha(original_raw),'issues':original['issues']},
        'inherited_artifact_checks':inherited,'source_receipts':source_receipts,'extra_sources':extra_sources,'changed_sources':changed,'diffs':diffs,
        'freeze':{'path':freeze_path,'sha256':sha(freeze_raw),'index_page':page,'home_matches_frozen_exactly':home_same},
        'inventory':{'count':len(index),'headings':len(headings),'same_ids_and_titles':True,'all_footer_links':True},
        'opening_execution':{'command':[sys.executable,'-c',opening],'stdout':opening_run.stdout,'stderr':opening_run.stderr,'exit_code':opening_run.returncode,'matches_original':True},
        'changed_sentence_checks':{
            'gradient_factor':gradients,
            'normalize':{'matrix_after_text_swap':similarity.tolist(),'selected_columns':similarity.argmax(-1).tolist(),'integer_error':integer_error},
            'allclose':{'difference':float((close_a-close_b).abs().max()),'atol':1e-5,'default_rtol':1e-5,'with_default_rtol':default_close,'with_zero_rtol':absolute_only,'criterion':'abs(a-b) <= atol + rtol*abs(b)'},
            'frame_overlap':{'window':400,'hop':160,'event_captured_with_overlap':[399,401],'overlapping_frames_containing_event':[s for s in range(0,800,160) if s<=399 and s+400>=401],'event_not_captured_despite_overlap':[159,459],'frames_containing_counterexample':[s for s in range(0,800,160) if s<=159 and s+400>=459]},
            'periodic_extension':periodic,'window_energy':energies,
            'identical_transcript_derivation':'只觀察同一字串 x 的規則，對兩個隱藏音高條件的輸出分布同為 P(answer|x)。固定規則不能給兩個不同確定答案；隨機規則也沒有辨識兩個條件的額外資訊。',
            'mean_log_features':mean_report,
            'supervised_order_target_derivation':'時間平均對排列不變；若只剩平均摘要，無法區分對應兩種先後。保留順序是輸入資訊條件；以先後答案監督才讓該訓練目標區分兩種先後，但不保證訓練必然成功。',
            'adaptive_pool':{'input_length':44,'output_length':16,'bins':bins,'neighboring_bins_overlap':any(bins[i][1]>bins[i+1][0] for i in range(15)),'manual_means_equal_torch':True,'encoder_shapes':encoded},
            'diagnostic_scope':'打字答對只核實文字輸入路徑，不能證明核心對語音接頭送來的不同特徵也正確；因此改為先查入口與接頭並確認核心能用該特徵，是合理排查方向，不是失敗原因的充分判定。'},
        'same_language_backbone_execution':{'shares_language_object':multimodal.language is language,'text_wrapper_exactly_equal':bool(torch.equal(direct,wrapped)),'joint_image_audio_logits_shape':list(both.shape)},
        'public_resources':public,'public_build_info':info,'public_version_and_resources_unchanged':exact_public_version,
        'limitations':['未重跑全部教材、長時間訓練或模型品質評估。','手機導覽沿用原始 HTML/CSS 核對，未做手機或瀏覽器點擊測試；Colab 只核對入口。','公開網站完整逐頁與搜尋執行證據沿用原報告，並重新讀取首頁、搜尋索引、worker、CSS 與 build-info 核對版本。']}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

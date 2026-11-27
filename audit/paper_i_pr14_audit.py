#!/usr/bin/env python3
"""PR14 clean-checkout audit. No scientific training, checkpoint loads or protected evaluation."""
import ast, collections, csv, hashlib, importlib.util, itertools, json, math, os, re, statistics, subprocess, sys
from pathlib import Path
BASE='8f2add239c878e312553362c752c9da8e396cbd7'
CANDIDATE='910c320a3696be4a4807ef554602adff3965d091'
def git(*args): return subprocess.check_output(['git',*args], text=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def canon(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check_close(a,b): return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def records(group,pattern): return sorted(Path('reports',group).glob(pattern))
def report_mean(r): return statistics.mean(v['C_query'] for x in r['native'].values() for v in x.values())
def walk(x, path=()):
    if isinstance(x,dict):
        for k,v in x.items(): yield from walk(v,path+(str(k),))
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from walk(v,path+(str(i),))
    else: yield path,x
errors=[]; warnings=[]; result={}
changes=[x.split('\t') for x in git('diff','--name-status',BASE,CANDIDATE).splitlines()]
paths=[p for status,p in changes]
assert len(paths)==1479 and all(s=='A' for s,p in changes)
assert git('diff','--name-only',CANDIDATE,'HEAD').splitlines()==['.github/workflows/paper-i-pr14-audit.yml','audit/paper_i_pr14_audit.py']
assert not git('diff','--name-only',CANDIDATE,'--',*paths)
tracked=git('ls-tree','-r','--name-only',CANDIDATE).splitlines()
hashes={p:sha(p) for p in tracked}
byhash=collections.defaultdict(list)
for p,h in hashes.items(): byhash[h].append(p)
result['tree']={'added':len(paths),'bytes':sum(Path(p).stat().st_size for p in paths),'all_existing_preserved':True}
py=[p for p in paths if p.endswith('.py')]; sh=[p for p in paths if p.endswith('.sh')]
imports=collections.defaultdict(set); syntax_errors=[]; wrapper_missing=[]; bash_invocation=[]
for p in py:
    try:
        tree=ast.parse(Path(p).read_text(),filename=p)
        compile(tree,p,'exec')
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):
                for a in n.names: imports[a.name.split('.')[0]].add(p)
            elif isinstance(n,ast.ImportFrom) and n.module and not n.level: imports[n.module.split('.')[0]].add(p)
    except Exception as exc: syntax_errors.append((p,str(exc)))
for p in sh:
    q=subprocess.run(['bash','-n',p],capture_output=True,text=True)
    if q.returncode: syntax_errors.append((p,q.stderr))
    if not os.access(p,os.X_OK): bash_invocation.append(p)
    text=Path(p).read_text()
    for name in set(re.findall(r'(?:scripts/)?(?:run_)?[A-Za-z0-9_]+\.py\b',text)):
        if not (Path(name).is_file() or Path('scripts',name).is_file()): wrapper_missing.append((p,name))
sys.path[:0]=['scripts','src']
external={}; unresolved={}
for name,users in sorted(imports.items()):
    if name in sys.stdlib_module_names or name=='__future__': continue
    local=Path('scripts',name+'.py').is_file() or Path('src',name).is_dir() or Path('src',name+'.py').is_file()
    if not local:
        try: available=importlib.util.find_spec(name) is not None
        except Exception: available=False
        external[name]=available
        if not available: unresolved[name]=sorted(users)
result['syntax_dependencies']={'python':len(py),'shell':len(sh),'syntax_errors':syntax_errors,'unresolved_imports':unresolved,'external_packages':external,'wrapper_literal_paths_missing':wrapper_missing,'nonexecutable_wrappers_use_bash':bash_invocation}
errors.extend(syntax_errors); errors.extend(wrapper_missing)
manifests=[p for p in paths if p.startswith('reports/') and 'manifest' in Path(p).name and p.endswith('.json')]
assert len(manifests)==27
manifest_status=[]; named_hash_failures=[]; semantic_hashes=[]; digest_failures=[]; file_hash_matches=0; named_hash_checks=0; checkpoint_hash_refs=0
for p in manifests:
    j=load(p); pr=j.get('protocol',j); item={'path':p,'canonical_protocol_digest':'not declared','named_path_checks':0,'file_digest_matches':0,'unresolved_semantic_or_artifact_hashes':0}
    if 'protocol_digest' in j and 'protocol' in j:
        ok=canon(pr)==j['protocol_digest']; item['canonical_protocol_digest']=ok
        if not ok: digest_failures.append(p)
    for loc,v in walk(j):
        if not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v): continue
        k=loc[-1]; key='.'.join(loc)
        if k.endswith(('.py','.sh','.md','.json','.csv','.svg','.yml','.yaml')):
            poss=[k,'scripts/'+k,'paper/'+k]
            found=[x for x in poss if x in hashes]
            if len(found)==0: found=[x for x in tracked if Path(x).name==k]
            if len(found)==1:
                named_hash_checks+=1; item['named_path_checks']+=1
                if hashes[found[0]]!=v: named_hash_failures.append((p,key,found[0]))
                continue
            if len(found)>1 and any(hashes[x]==v for x in found):
                named_hash_checks+=1; item['named_path_checks']+=1; continue
            named_hash_failures.append((p,key,'missing or ambiguous named asset')); continue
        if v in byhash:
            file_hash_matches+=1; item['file_digest_matches']+=1
        else:
            item['unresolved_semantic_or_artifact_hashes']+=1
            if any(x in key.lower() for x in ('weights','checkpoint','ckpt','tensor')): checkpoint_hash_refs+=1
            if re.search(r'(script|launcher|protocol|generator|source_manifest|parent_summary|aggregate|evaluation|manifest)_(sha256|sha)$',k) and 'digest' not in k:
                semantic_hashes.append((p,key))
    manifest_status.append(item)
result['manifests']={'n':len(manifests),'canonical_protocol_digest_failures':digest_failures,'named_path_checks':named_hash_checks,'named_path_hash_failures':named_hash_failures,'other_hashes_matching_tracked_bytes':file_hash_matches,'unresolved_filelike_fields':semantic_hashes,'unmatched_checkpoint_or_tensor_hash_references':checkpoint_hash_refs,'detail':manifest_status}
errors.extend(digest_failures);errors.extend(named_hash_failures)
# Inspect all imported reports, without printing any source strings.
report_paths=[p for p in paths if p.startswith('reports/')]
raw_fields=collections.Counter(); raw_examples=collections.defaultdict(set); scope_flags=[]; parse_errors=[]; nonfinite=[]; digest_mismatch=[]; csv_rows=0; json_count=0; csv_count=0; md_count=0
raw_keys={'text','source_text','sentence','prompt','completion','generated_text','response','input_text','raw_text','passage','document','context','question_text'}
def scan(k,v,p):
    if isinstance(v,bool):
        low=k.lower()
        if ('protected' in low or 'reserved' in low) and ((low.startswith('no_') and not v) or (('accessed' in low or 'opened' in low or 'used' in low) and not low.startswith('no_') and v)):
            scope_flags.append((p,k,v))
    if isinstance(v,float) and not math.isfinite(v): nonfinite.append(p)
    if k.lower() in raw_keys and isinstance(v,str) and v.strip():
        raw_fields[k]+=1; raw_examples[k].add(p)
for p in report_paths:
    try:
        if p.endswith('.json'):
            j=load(p); json_count+=1
            for loc,v in walk(j):
                if loc: scan(loc[-1],v,p)
            parent=Path(p).parent
            for folder in [parent,*parent.parents]:
                mp=folder/'study_manifest.json'
                if mp.is_file() and str(mp)!=p:
                    m=load(mp)
                    if isinstance(j,dict) and 'protocol_digest' in j and 'protocol_digest' in m and j['protocol_digest']!=m['protocol_digest']: digest_mismatch.append(p)
                    break
                if folder==Path('reports'): break
        elif p.endswith('.csv'):
            csv_count+=1
            with open(p,newline='',encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    csv_rows+=1
                    if None in row: raise ValueError('CSV extra columns')
                    for k,v in row.items(): scan(k,v,p)
        elif p.endswith('.md'):
            md_count+=1;Path(p).read_text(encoding='utf-8')
        else: parse_errors.append((p,'unexpected extension'))
    except Exception as e: parse_errors.append((p,str(e)))
result['reports']={'files':len(report_paths),'json':json_count,'csv':csv_count,'markdown':md_count,'csv_rows':csv_rows,'parse_errors':parse_errors,'report_manifest_digest_mismatch':digest_mismatch,'nonfinite_files':sorted(set(nonfinite)),'scope_flags_requiring_review':scope_flags,'raw_text_field_counts':dict(raw_fields),'raw_text_file_examples':{k:sorted(v)[:8] for k,v in raw_examples.items()},'rights_limit':'Automated screening does not establish rights clearance or absence of sensitive content. Manual provenance review remains required.'}
errors.extend(parse_errors);errors.extend(digest_mismatch)
expect=[
('core_diversity_v1','seed*_step6400.json',16),('train_surface_factorial_v0','seed*_step6400.json',12),('train_surface_cross_eval_v0','seed*_cross_eval.json',12),('train_role_anchor_probe_v0','seed*_role_anchor.json',12),('role_diversity_complex_v0','seed*_step6400.json',16),('role_diversity_recovery_v0','seed*_step*.json',16),('role_circuit_diagnostics_v0','per_checkpoint/*.json',16),('role_representation_localisation_v0','per_checkpoint/*.json',16),('role_causal_routing_v0','per_checkpoint/*.json',16),('role_causal_pathways_v0','per_checkpoint/*.json',16),('role_binding_causal_validation_v0','per_checkpoint/*.json',16),('role_binding_localisation_v0','per_checkpoint/*.json',16),('role_mechanism_replication_v0','per_checkpoint/*.json',64),('role_mechanism_discovery_v1','per_checkpoint/*.json',66),('role_mechanism_repair_v0','per_arm/*.json',168),('role_repair_specificity_v1','per_arm/*.json',384),('position_cb_core_transfer_v0','seed*_core_audit.json',16)]
complete={}
for group,pattern,n in expect:
    found=len(records(group,pattern));complete[group]={'observed':found,'expected':n,'pass':found==n}
    if found!=n: warnings.append((group,'missing or extra reports',found,n))
result['completeness']=complete
# Independent read-only recomputations of central numerical claims.
recomputed={}
def verify(name,fn):
    try: recomputed[name]={'pass':True,'value':fn()}
    except Exception as exc:
        recomputed[name]={'pass':False,'error':repr(exc)};errors.append((name,repr(exc)))
def surface():
    group='train_surface_factorial_v0';s=load(Path('reports',group,'analysis_summary.json'));rr=[load(p) for p in records(group,'seed*_step6400.json')]
    assert len(rr)==12
    for pop,arms in s['arm_metrics'].items():
        for arm,metrics in arms.items():
            for k in ('C_query','accuracy4','full_vocab_accuracy'):
                vs=[statistics.mean(v[k] for v in r['outcomes'][arm][pop].values()) for r in rr]
                assert check_close(statistics.mean(vs),metrics[k]['mean'])
    def contrast(r,pop):
        def m(a): return statistics.mean(v['C_query'] for v in r['outcomes'][a][pop].values())
        return .5*((m('bare')-m('adjunct_only'))+(m('frame_only')-m('frame_adjunct')))
    vs=[contrast(r,'unseen_roles') for r in rr];assert check_close(statistics.mean(vs),s['primary']['estimate']['mean']);assert sum(x>0 for x in vs)==9
    return {'mean':statistics.mean(vs),'positive':sum(x>0 for x in vs)}
def diversity():
    group='role_diversity_complex_v0';s=load(Path('reports',group,'analysis_summary.json'));rr=[load(p) for p in records(group,'seed*_step6400.json')]
    assert len(rr)==16
    vs=[statistics.mean(v['C_query'] for v in r['outcomes']['roles_8']['unseen_roles_rich'].values())-statistics.mean(v['C_query'] for v in r['outcomes']['roles_1']['unseen_roles_rich'].values()) for r in rr]
    assert check_close(statistics.mean(vs),s['primary']['estimate']['mean']);assert sum(v>0 for v in vs)==0
    return {'mean':statistics.mean(vs),'positive':0}
def diagnostic():
    group='role_circuit_diagnostics_v0';s=load(Path('reports',group,'analysis_summary.json'));groups=collections.defaultdict(list)
    for p in records(group,'per_checkpoint/*.json'):
        for r in load(p)['rows']:
            groups[tuple(r[k] for k in ('seed','step','arm','population'))].append({**r['metrics'],**r['native']})
    assert len(groups)==448 and all(len(v)==2 for v in groups.values())
    for r in s['voice_averaged']:
        v=groups[tuple(r[k] for k in ('seed','step','arm','population'))]
        for k,x in r.items():
            if k not in ('seed','step','arm','population'):assert check_close(statistics.mean(a[k] for a in v),x)
    return {'voice_averaged_groups':len(groups),'raw_rows':sum(map(len,groups.values()))}
def replication():
    group='role_mechanism_replication_v0';s=load(Path('reports',group,'analysis_summary.json'));rr=[load(p) for p in records(group,'per_checkpoint/*.json')];idx={(r['seed'],r['step']):r for r in rr}
    assert len(idx)==64
    for q in s['per_seed']:
        r=idx[q['seed'],q['step']];assert q['category']==r['category']
        assert q['native_joint']=={p:{v:x['C_query'] for v,x in a.items()} for p,a in r['native'].items()}
    final=[r for r in rr if r['step']==51200];counts=collections.Counter(r['category'] for r in final)
    success=sum(all(v['C_query']>=.95 for a in r['native'].values() for v in a.values()) for r in final)
    assert success==s['n_successful']==9 and all(counts[k]==v for k,v in s['counts'].items())
    return {'checkpoints':len(rr),'success':success,'zero':sum(report_mean(r)==0 for r in final)}
def repair():
    group='role_repair_specificity_v1';s=load(Path('reports',group,'analysis_summary.json'));rr=[load(p) for p in records(group,'per_arm/*.json')];idx={(r['run'],r['arm'],r['step']):r for r in rr};assert len(idx)==384
    assert all(r['audit']['frozen_max']==0 for r in rr)
    for k,c in s['contrasts'].items():
        step,ab=k.split(':');a,b=ab.split('-');v=[report_mean(idx[int(run),a,int(step)])-report_mean(idx[int(run),b,int(step)]) for run in c['per_run']]
        assert all(check_close(x,y) for x,y in zip(v,c['per_run'].values()))
        assert check_close(statistics.mean(v),c['mean'])
    for a,runs in s['first_observed_recovery'].items():
        for run,t in runs.items():
            found=next((t2 for t2 in (0,400,800,1600,3200,6400) if report_mean(idx[int(run),a,t2])>=.95),None);assert found==t
    return {'reports':len(rr),'primary':s['primary']['mean']}
for name,fn in [('surface',surface),('diversity',diversity),('diagnostics',diagnostic),('replication',replication),('repair_specificity',repair)]:verify(name,fn)
result['recomputed']=recomputed
result['errors']=errors;result['warnings']=warnings
Path('/tmp/paper-i-pr14-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
for k in ('tree','syntax_dependencies','manifests','reports','completeness','recomputed','errors','warnings'):
    v=result[k]
    if k=='manifests':v={k2:x for k2,x in v.items() if k2!='detail'}
    print(k,json.dumps(v,sort_keys=True),flush=True)
# Fail on real validation errors; separate known gaps and unresolved rights from successful checks.
sys.exit(bool(errors))

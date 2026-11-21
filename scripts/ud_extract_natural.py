import glob,collections,re
PERSON_VERBS=set("""help call meet ask thank hire fire kill attack arrest marry love hate see visit follow beat support invite accuse warn pay join
praise blame push pull kick hit punch slap hug kiss greet watch chase rescue save protect defend attack threaten interview contact
email phone tell encourage convince persuade teach train hurt injure shoot stab murder bite catch trust admire respect like dislike
fear miss leave abandon betray replace defeat elect appoint promote nominate sue fight date recognize remember forget meet introduce
congratulate criticize insult mock tease scold punish reward thank comfort feed bother annoy surprise frighten scare impress inform
notify hire employ recruit represent serve treat examine kidnap rob cheat fool trick hold grab lift carry drive take bring send""".split())
def sents(f):
    s=[];meta={};mwt={}
    for line in open(f,encoding='utf8'):
        line=line.rstrip('\n')
        if not line:
            if s: yield meta,s
            s=[];meta={};mwt={}
        elif line.startswith('#'):
            if '=' in line: k,v=line[2:].split('=',1); meta[k.strip()]=v.strip()
        else:
            c=line.split('\t')
            if '.' in c[0]: continue
            if '-' in c[0]:  # multiword token ("don't" = do + n't): no space inside, range's SpaceAfter at its end
                lo,hi=(int(x) for x in c[0].split('-')); mwt[hi]=c[9]; mwt.update({k:'SpaceAfter=No' for k in range(lo,hi)})
                continue
            if int(c[0]) in mwt: c[9]=mwt.pop(int(c[0]))
            s.append(c)
def subtree(s,head):
    ids={head[0]}; ch=True
    while ch:
        ch=False
        for c in s:
            if c[6] in ids and c[0] not in ids: ids.add(c[0]); ch=True
    idx=sorted(int(i) for i in ids)
    return idx if idx==list(range(idx[0],idx[-1]+1)) else None
def extract(files):
    out=[]
    for f in files:
        for meta,s in sents(f):
            for v in s:
                if v[3]!='VERB' or v[2].lower() not in PERSON_VERBS: continue
                deps=[c for c in s if c[6]==v[0]]
                one=lambda rel:[c for c in deps if c[7]==rel]
                pas='Pass' in v[5] or one('aux:pass')
                if not pas and len(one('nsubj'))==1 and len(one('obj'))==1:
                    a,p=one('nsubj')[0],one('obj')[0]; kind='active'
                elif pas and len(one('nsubj:pass'))==1 and len(one('obl:agent'))==1:
                    p,a=one('nsubj:pass')[0],one('obl:agent')[0]; kind='passive'
                else: continue
                if a[3] not in('NOUN','PROPN') or p[3] not in('NOUN','PROPN'): continue
                sa,sp=subtree(s,a),subtree(s,p)
                if not sa or not sp: continue
                if kind=='passive': sa=[i for i in sa if s[i-1][7]!='case'] # drop "by"
                if not sa or len(sa)>6 or len(sp)>6: continue
                out.append(dict(file=f,sent_id=meta.get('sent_id'),kind=kind,lemma=v[2].lower(),tokens=[c[1] for c in s],
                                spaces=['SpaceAfter=No' not in c[9] for c in s],a=sa,p=sp,verb=int(v[0])))
    return out
if __name__=='__main__':
    o=extract(sorted(glob.glob('*.conllu')))
    print(len(o),collections.Counter(x['kind'] for x in o)); print(collections.Counter(x['lemma'] for x in o).most_common(25))
    import random
    for x in random.Random(1).sample(o,8):
        t=x['tokens']; print(x['kind'],'|',' '.join(t[i-1] for i in x['a']),'|',x['lemma'],'|',' '.join(t[i-1] for i in x['p']),'||',' '.join(t)[:160])

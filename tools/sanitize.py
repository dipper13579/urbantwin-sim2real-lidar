"""Scan the complete release tree and every reachable Git blob/commit.

Findings disclose only relative filename, line and category, never matched values.
Git object storage is decoded via Git; generated index/pack bytes are not text.
"""
import argparse,json,re,subprocess
from pathlib import Path
RULES={
 'user_absolute_path':re.compile(r'(?:/'+r'home/[^/\s"\']+|[A-Za-z]:[/\\]'+r'Users[/\\][^/\\\s"\']+|/mnt/[a-z]/'+r'Users/[^/\s"\']+)'),
 'email':re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
 'ipv4':re.compile(r'(?<![\w.])(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}(?![\w.])'),
 'private_key':re.compile('-----BEGIN '+r'(?:RSA |EC |OPENSSH )?'+'PRIVATE KEY-----'),
 'provider_token':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{10,})'),
 'assigned_secret':re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|cookie)\s*[:=]\s*["\'][A-Za-z0-9_./+=-]{16,}["\']'),
 'bearer':re.compile(r'(?i)Bearer\s+[A-Za-z0-9_.=-]{20,}'),
}
FORBIDDEN_EXT={'.pth','.pt','.ckpt','.bin','.npy','.npz','.pkl','.pickle','.pcd','.ply','.zip','.7z','.tar','.gz','.pdf'}
def scan(root):
    findings=[];counts={k:0 for k in RULES};counts.update(oversize=0,forbidden_file=0,symlink=0)
    def hit(name,category,line=0):
        counts[category]+=1;findings.append({'file':name,'line':line,'category':category})
    def content(name,raw):
        text=raw.decode('utf-8',errors='replace')
        for category,pattern in RULES.items():
            for m in pattern.finditer(text):hit(name,category,text.count('\n',0,m.start())+1)
    files=[]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if '.git' in rel.parts:continue
        if p.is_symlink():hit(str(rel),'symlink');continue
        if not p.is_file():continue
        files.append(str(rel))
        if p.stat().st_size>50_000_000:hit(str(rel),'oversize')
        if p.suffix.lower() in FORBIDDEN_EXT or p.name=='.env' or p.name.startswith('.env.') or p.name.lower() in ['cookies','cookies.sqlite','login data','web data']:
            hit(str(rel),'forbidden_file')
        content(str(rel),p.read_bytes())
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args])
    objects=git('rev-list','--objects','--all').decode().splitlines();blobs=0;commits=0
    for row in objects:
        oid=row.split(' ',1)[0];kind=git('cat-file','-t',oid).strip()
        if kind not in [b'blob',b'commit',b'tag']:continue
        raw=git('cat-file','-p',oid);name='git:'+row
        content(name,raw)
        if kind==b'blob':
            blobs+=1
            if len(raw)>50_000_000:hit(name,'oversize')
            path=Path(row.split(' ',1)[1]) if ' ' in row else Path('')
            if path.suffix.lower() in FORBIDDEN_EXT:hit(name,'forbidden_file')
        elif kind==b'commit':commits+=1
    # Git config, hooks and reflogs can contain identifiers even when blobs are clean.
    for p in sorted((root/'.git').rglob('*')):
        if not p.is_file() or 'objects' in p.relative_to(root/'.git').parts or p.name in ['index','COMMIT_EDITMSG']:continue
        content(str(p.relative_to(root)),p.read_bytes())
    remotes=git('remote').decode().splitlines()
    return {'passed':not findings and not remotes,'files_scanned':len(files),'reachable_blobs_scanned':blobs,'reachable_commits_scanned':commits,'hits_by_category':counts,'total_hits':len(findings),'findings':findings,'remotes':remotes,'scope':'Every worktree file, reachable Git blob/commit/tag, Git config/hooks/reflogs. Object packs decoded with git cat-file. No path/content allowlist.','limitations':'Rule-based scanning is not a proof that arbitrary unknown credentials cannot exist. No raw match values printed.'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path);a=ap.parse_args()
    result=scan(a.root.resolve());text=json.dumps(result,indent=2)
    if a.out:a.out.write_text(text+'\n')
    print(text);raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()

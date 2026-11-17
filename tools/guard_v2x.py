"""Fail closed unless inputs match the canonical-identity audit; no real IDs stored here."""
import hashlib,json
from release_paths import REPO,RAW_ROOT,KIT_ROOT
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify():
    e=json.loads((REPO/'provenance/v2x_validation_guard.json').read_text())
    if sha(RAW_ROOT/'V2X-Real/val.zip')!=e['archive_sha256']:raise ValueError('Unaudited V2X archive; repeat original scenario identity audit first')
    if sha(KIT_ROOT/'v2x/forbidden_frames.txt')!=e['forbidden_list_sha256']:raise ValueError('Changed official forbidden list; repeat identity audit first')
    return e
if __name__=='__main__':print(json.dumps(verify(),indent=2))

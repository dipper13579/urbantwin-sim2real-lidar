"""Compare every ordered record and byte against predictions.json or a submission ZIP."""
import argparse,json,hashlib,zipfile
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--actual',type=Path,required=True);ap.add_argument('--expected',type=Path,required=True);a=ap.parse_args()
    actual=a.actual.read_bytes()
    if a.expected.suffix=='.zip':
        with zipfile.ZipFile(a.expected) as z:expected=z.read('predictions.json')
    else:expected=a.expected.read_bytes()
    x,y=json.loads(actual),json.loads(expected)
    result={'frames':len(x),'boxes':sum(map(len,x.values())),'frame_ids_equal':set(x)==set(y),'ordered_records_equal':x==y,'byte_equal':actual==expected,'sha256':hashlib.sha256(actual).hexdigest(),'mismatch_frame_count':sum(x.get(f)!=y.get(f) for f in set(x)|set(y))}
    print(json.dumps(result,indent=2));raise SystemExit(0 if x==y else 1)
if __name__=='__main__':main()

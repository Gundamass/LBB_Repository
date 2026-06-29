#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

VALID = {"plain particle", "dirt", "scratch", "collision"}

def norm_box(b):
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x1,y1,x2,y2 = map(float, b)
    except Exception:
        return None
    x1,x2 = sorted((x1,x2)); y1,y2 = sorted((y1,y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1,y1,x2,y2]

def iou(a,b):
    a=norm_box(a); b=norm_box(b)
    if a is None or b is None: return 0.0
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    den=aa+bb-inter
    return inter/den if den>0 else 0.0

class Store:
    def __init__(self, p: Path):
        self.p=p
        self.z=None
        self.members={}
        if p.suffix.lower()=='.zip':
            self.z=zipfile.ZipFile(p)
            self.members={Path(n).name:n for n in self.z.namelist() if n.endswith('.json')}
    def read(self, name):
        try:
            if self.z:
                m=self.members.get(name)
                if not m: return None
                return json.loads(self.z.read(m).decode('utf-8'))
            fp=self.p/name
            if not fp.exists(): return None
            return json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            return None
    def close(self):
        if self.z: self.z.close()

def load_gt(gt_dir: Path):
    rows=[]
    for p in sorted(gt_dir.rglob('*.json')):
        try:
            data=json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(data.get('annotations'), list):
            image_id = Path(str(data.get('image_id') or p.with_suffix('.jpg').name)).name
            for a in data.get('annotations', []):
                if not isinstance(a, dict):
                    continue
                label=str(a.get('label',''))
                if label not in VALID:
                    continue
                box=norm_box(a.get('bbox', []))
                if box:
                    rows.append((image_id, p.name, label, box))
            continue
        image_id = data.get('imagePath') or p.with_suffix('.jpg').name
        image_id = Path(image_id).name
        for s in data.get('shapes', []):
            label=str(s.get('label',''))
            if label not in VALID: continue
            pts=s.get('points', [])
            if len(pts)<2: continue
            try:
                xs=[float(x[0]) for x in pts]; ys=[float(x[1]) for x in pts]
            except Exception:
                continue
            box=[min(xs), min(ys), max(xs), max(ys)]
            if norm_box(box):
                rows.append((image_id, p.with_suffix('.json').name, label, box))
    return rows

def preds_for(store: Store, image_id: str, score_thr: float):
    payload = store.read(Path(image_id).with_suffix('.json').name)
    if payload is None:
        # Some payloads use Bxxx.json names matching image id; fallback is same anyway.
        return []
    rows=[]
    for a in payload.get('annotations', []):
        if not isinstance(a, dict): continue
        label=str(a.get('label',''))
        if label not in VALID: continue
        try: score=float(a.get('confidence',0.0) or 0.0)
        except Exception: score=0.0
        if score < score_thr: continue
        box=norm_box(a.get('bbox', []))
        if box: rows.append((label, score, box))
    return rows

def best_hit(pred_rows, label, box):
    best=(0.0, 0.0, None)
    any_label=(0.0, 0.0, None)
    for pl, sc, pb in pred_rows:
        v=iou(pb, box)
        if v > any_label[0]: any_label=(v, sc, pl)
        if pl == label and v > best[0]: best=(v, sc, pl)
    return best, any_label

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gt-dir', default='./手标数据')
    ap.add_argument('--base', required=True)
    ap.add_argument('--branch', action='append', default=[], help='name=zip')
    ap.add_argument('--score-thr', type=float, default=0.0)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--out-json', default='outputs/branch_unique_hits.json')
    args=ap.parse_args()
    branches=[]
    for item in args.branch:
        if '=' in item:
            name,path=item.split('=',1)
        else:
            path=item; name=Path(path).stem[:60]
        branches.append((name, Path(path)))
    stores=[('base', Store(Path(args.base)))] + [(n, Store(p)) for n,p in branches]
    gts=load_gt(Path(args.gt_dir))
    print('GT objects:', len(gts), Counter(x[2] for x in gts))
    hit=Counter(); hit_by_label=defaultdict(Counter); base_miss_branch_hit=Counter(); base_miss_branch_hit_by_label=defaultdict(Counter)
    oracle_hit=0; base_hit=0
    examples=defaultdict(list)
    for image_id, gt_json, glabel, gbox in gts:
        per=[]
        for name, st in stores:
            pr=preds_for(st, image_id, args.score_thr)
            same, anyl=best_hit(pr, glabel, gbox)
            ok=same[0] >= args.iou_thr
            per.append((name, ok, same, anyl))
            if ok:
                hit[name]+=1; hit_by_label[name][glabel]+=1
        b_ok=per[0][1]
        if b_ok: base_hit += 1
        if any(x[1] for x in per): oracle_hit += 1
        if not b_ok:
            for name, ok, same, anyl in per[1:]:
                if ok:
                    base_miss_branch_hit[name]+=1
                    base_miss_branch_hit_by_label[name][glabel]+=1
                    if len(examples[name]) < 20:
                        examples[name].append({
                            'image_id': image_id, 'gt_json': gt_json, 'label': glabel, 'gt_box': [round(v,2) for v in gbox],
                            'branch_iou': round(float(same[0]),4), 'branch_score': round(float(same[1]),8),
                        })
    for _,st in stores: st.close()
    summary={
        'gt_total': len(gts),
        'gt_by_label': dict(Counter(x[2] for x in gts)),
        'base_hit': base_hit,
        'oracle_hit_any_branch': oracle_hit,
        'hit': dict(hit),
        'hit_by_label': {k:dict(v) for k,v in hit_by_label.items()},
        'base_miss_branch_hit': dict(base_miss_branch_hit),
        'base_miss_branch_hit_by_label': {k:dict(v) for k,v in base_miss_branch_hit_by_label.items()},
        'examples': examples,
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['base_hit','oracle_hit_any_branch','hit','base_miss_branch_hit','base_miss_branch_hit_by_label']}, ensure_ascii=False, indent=2))
    print('Saved:', args.out_json)

if __name__ == '__main__':
    main()

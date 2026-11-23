"""One fixed, pinned data4 download job; no training or robot actions."""
import os,json,time,hashlib
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
os.environ['HF_HUB_ETAG_TIMEOUT']='30'
from huggingface_hub import snapshot_download
root=Path('/home/data4/hts/demo1/fr5-learning')
revision='22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0'
sha_expected='0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'
status=root/'logs/download-status.json'
def save(x):
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(status)
try:
 save({'stage':'base_weights_downloading','revision':revision,'started_unix':time.time()})
 p=Path(snapshot_download('lerobot/pi05_base',revision=revision,local_dir=root/'models/pi05_base',token=False,max_workers=1))
 m=p/'model.safetensors'
 if m.stat().st_size!=14467165872:raise ValueError('Incomplete checkpoint size')
 h=hashlib.sha256()
 with m.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=sha_expected:raise ValueError('Checkpoint hash mismatch')
 r={'repo':'lerobot/pi05_base','revision':revision,'path':str(p),'weights_bytes':m.stat().st_size,'weights_sha256':h.hexdigest(),'completed_at':time.time()}
 (root/'models/pi05-download-manifest.json').write_text(json.dumps(r,indent=2))
 save({'stage':'base_complete_acquiring_authorized_tokenizer','revision':revision})
 # Uses the user's existing credential and ordinary gated access checks.
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224',revision='35e4f46485b4d07967e7e9935bc3786aad50687c')
 tok.save_pretrained(root/'models/paligemma-tokenizer')
 save({'stage':'base_and_tokenizer_ready','revision':revision,'completed_unix':time.time(),'training_started':False})
except Exception as e:
 save({'stage':'download_stopped','error_class':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None),'stopped_unix':time.time(),'training_started':False})
 raise SystemExit(type(e).__name__)

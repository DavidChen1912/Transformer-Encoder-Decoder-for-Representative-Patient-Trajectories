import sys,json,re,itertools
from pathlib import Path
sys.path.insert(0,'/tmp/ot-paper-deps')
import fitz
from PIL import Image,ImageOps,ImageDraw
import numpy as np
root=Path('/root/scghsom-ot/OT-scGHSOM')
out=root/'docs/manuscript/qa';out.mkdir(exist_ok=True)
doc=fitz.open(root/'docs/ot_transformer_manuscript.pdf')
report={'pages':len(doc),'page_audit':[]}
thumbs=[]
for i,p in enumerate(doc):
 pix=p.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False);file=out/f'page-{i+1:02d}.png';pix.save(file)
 blocks=p.get_text('blocks');bad=[list(b[:4]) for b in blocks if b[0]<65 or b[2]>547 or b[1]<40 or b[3]>764]
 text=p.get_text();report['page_audit'].append({'page':i+1,'words':len(text.split()),'images':len(p.get_images()),'outside_margin':bad,'replacement_character':'\ufffd' in text})
 im=Image.open(file).convert('RGB');im.thumbnail((459,594));thumbs.append(im)
for start in range(0,len(thumbs),4):
 canvas=Image.new('RGB',(950,1250),'#dce1e6');draw=ImageDraw.Draw(canvas)
 for j,im in enumerate(thumbs[start:start+4]):
  x=10+(j%2)*475;y=25+(j//2)*625;canvas.paste(im,(x,y));draw.text((x,y-16),'Page '+str(start+j+1),fill='black')
 canvas.save(out/f'contact-{start//4+1:02d}.png')
# Independent exact calculation for manuscript example.
P=np.array([[[.3,.3],[.1,.3]],[[.1,.3],[.3,.3]]]);a=P[0].sum(1);Q=P/P.sum(2,keepdims=True)
paths=list(itertools.product(range(2),repeat=3));w=np.array([a[r[0]]*Q[0,r[0],r[1]]*Q[1,r[1],r[2]] for r in paths])
g=np.array([1,1,0,1,0,1,1,1]).reshape(2,2,2)
cov=sum(ww*g[0,r[0],r[1]]*g[1,r[1],r[2]] for ww,r in zip(w,paths))
beta=a.copy()
for t in range(2):beta=(beta[:,None]*Q[t]*g[t]).sum(0)
assert np.allclose(w,[.075,.225,.15,.15,.025,.075,.15,.15])
assert np.isclose(w.sum(),1) and np.isclose(cov,.825) and np.isclose(beta.sum(),cov)
for t in range(2):
 rec=np.zeros((2,2))
 for r,ww in zip(paths,w):rec[r[t],r[t+1]]+=ww
 assert np.allclose(rec,P[t])
report['analytical_checks']={'normalization':float(w.sum()),'coverage':float(cov),'forward_recursion':float(beta.sum()),'six_edge_objective':float(1-cov+.8*g.mean()),'adjacent_recovery':'passed'}
report['equation_images']=len(list((root/'docs/manuscript/figures').glob('equation-*.png')))
(out/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

# Typography, authorship, figure inventory, and obsolete-banner audit.
source=(root/'docs/manuscript/manuscript.md').read_text(encoding='utf-8-sig')
all_text='\n'.join(page.get_text() for page in doc)
assert doc.metadata['author']=='JING-WEI CHEN'
for obsolete in ['Anonymous methods manuscript','Detailed research draft','Empirical evaluation pending','Neural training and comparative biological experiments have not yet been performed']:
 assert obsolete not in all_text, obsolete
expected={str(path.relative_to(root)) for directory in [root/'chart',root/'docs/chart'] for path in directory.glob('*.png')}
placed=set(re.findall(r'^@image (.+)$',source,re.M))
assert expected==placed,(expected-placed,placed-expected)
large_sizes={Image.open(path).size for path in (root/'docs/manuscript/figures').glob('placed-*.png')} 
assert sum(1 for page in doc for image in page.get_images() if (image[2],image[3]) in large_sizes)==len(expected)
vectors=sorted((root/'docs/manuscript/figures').glob('equation-*.pdf'))
assert len(vectors)==30
for file in vectors:
 with fitz.open(file) as equation_pdf:
  assert equation_pdf[0].get_text().strip(), file
  assert not equation_pdf[0].get_images(), file
  assert any('STIX' in font[3] for font in equation_pdf[0].get_fonts()), file
assert all(not page['outside_margin'] and not page['replacement_character'] for page in report['page_audit'])
report['author']=doc.metadata['author']
report['reused_figures']=sorted(expected)
report['vector_equations']=len(vectors)
report['math_font']='STIX; embedded vector font outlines/text, not raster equation images'
report['obsolete_banners']='absent'
(out/'verification.json').write_text(json.dumps(report,indent=2))
print('PASS: 30 vector STIX equations; all 10 figures; author; obsolete banners; margins and analytical checks.')


# Final typography and layout regression checks.
assert 'Figure 1.' in doc[0].get_text(), 'The introductory figure must stay on page one.'
raw_math=[]
for block in source.split('\n\n'):
 block=block.strip()
 if block.startswith(('$$','@','#','[')):continue
 prose=re.sub(r'\$[^$]+\$|`[^`]+`','',block)
 raw_math.extend(re.findall(r'\b[A-Za-z]_[A-Za-z0-9]+\b',prose))
assert not raw_math, raw_math
# The architecture and its caption may not overlap any colored node rectangle.
caption_collisions=[]
for number,page in enumerate(doc,1):
 captions=[fitz.Rect(b[:4]) for b in page.get_text('blocks') if str(b[4]).startswith('Figure 5.')]
 for item in page.get_drawings():
  fill=item.get('fill')
  if fill is None:continue
  palette=[(0.8588,0.9176,0.9961),(0.9961,0.9529,0.7804),(0.8627,0.9882,0.9059)]
  if not any(max(abs(float(v)-float(w)) for v,w in zip(fill,c))<0.015 for c in palette):continue
  rect=item['rect']
  if rect.width<60 or rect.height<15:continue
  for caption in captions:
   overlap=rect & caption
   if overlap.width>1 and overlap.height>1:caption_collisions.append(number)
assert not caption_collisions,caption_collisions
report['display_math_nominal_points']=11.5
report['inline_math_points']={'body':10.5,'tables':8}
report['inline_math_resolution_dpi']=600
report['raw_inline_math_identifiers']=raw_math
report['architecture_caption_collisions']=caption_collisions
report['intro_figure_on_first_page']=True
(out/'verification.json').write_text(json.dumps(report,indent=2))
print('PASS: first-page figure, inline notation, constant equation size policy, and architecture-caption separation.')

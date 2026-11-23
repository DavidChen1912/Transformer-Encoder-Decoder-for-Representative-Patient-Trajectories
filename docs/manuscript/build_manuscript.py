from pathlib import Path
import os,re,math,sys
sys.path.append('/tmp/ot-paper-deps')
os.environ['MPLCONFIGDIR']='/tmp/ot-paper-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import Flowable,SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether,CondPageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing,Rect,Line,String,Polygon,Circle,Group
from reportlab.graphics import renderPDF
from xml.sax.saxutils import escape
from PIL import Image as PILImage
import hashlib,numpy as np
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'STIXGeneral','pdf.fonttype':42,'ps.fonttype':42})
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ASSETS=HERE/'figures';ASSETS.mkdir(exist_ok=True)
OUT=ROOT/'docs'/'ot_transformer_manuscript.pdf'
FONTROOT=Path(matplotlib.get_data_path())/'fonts'/'ttf'
for name,f in [('Body','STIXGeneral'),('Bold','STIXGeneralBol'),('Italic','STIXGeneralItalic')]:
 pdfmetrics.registerFont(TTFont(name,str(FONTROOT/(f+'.ttf'))))
pdfmetrics.registerFont(TTFont('Sans','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic',boldItalic='Bold')
styles={
'p':ParagraphStyle('p',fontName='Body',fontSize=10.5,leading=13.6,alignment=4,spaceAfter=6,autoLeading='max',allowWidows=0,allowOrphans=0),
'h1':ParagraphStyle('h1',fontName='Bold',fontSize=13,leading=17,spaceBefore=11,spaceAfter=6,keepWithNext=True),
'h2':ParagraphStyle('h2',fontName='Bold',fontSize=10.5,leading=14,spaceBefore=8,spaceAfter=5,keepWithNext=True),
'caption':ParagraphStyle('caption',fontName='Body',fontSize=8.5,leading=10.5,spaceAfter=8),
'cell':ParagraphStyle('cell',fontName='Body',fontSize=8,leading=10.5),
 'title':ParagraphStyle('title',fontName='Bold',fontSize=19,leading=24,alignment=1,spaceAfter=12),
'sub':ParagraphStyle('sub',fontName='Body',fontSize=12,leading=16,alignment=1,spaceAfter=16)}
story=[];eqcount=0

class VectorEquation(Flowable):
 def __init__(self,file,maxwidth=420):
  Flowable.__init__(self)
  self.xobj=pagexobj(PdfReader(str(file)).pages[0])
  box=list(map(float,self.xobj.BBox));self.origw=box[2]-box[0];self.origh=box[3]-box[1]
  # Constant 11.5 pt display mathematics; over-wide equations require explicit line breaks.
  self.scale=1;self.width=self.origw;self.height=self.origh
  if self.width>maxwidth:raise ValueError(f'Equation needs line breaking: {file}, width={self.width}')
 def draw(self):
  self.canv.saveState();self.canv.scale(self.scale,self.scale)
  self.canv.doForm(makerl(self.canv,self.xobj));self.canv.restoreState()

def equation(s):
 global eqcount
 eqcount+=1
 fig=plt.figure(figsize=(8,.7));fig.patch.set_alpha(0)
 tx=fig.text(.5,.5,'$'+s+'$',ha='center',va='center',fontsize=11.5,color='black')
 fig.canvas.draw();bbox=tx.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.025,1.2)
 file=ASSETS/f'equation-{eqcount:02d}.pdf'
 fig.savefig(file,bbox_inches=bbox,transparent=True,pad_inches=.035)
 fig.savefig(file.with_suffix('.png'),dpi=350,bbox_inches=bbox,transparent=True,pad_inches=.035);plt.close(fig)
 t=Table([[VectorEquation(file),Paragraph(f'({eqcount})',styles['cell'])]],colWidths=[432,36])
 t.setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(-1,-1),'CENTER'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
 story.append(t)

def existing_image(path):
 file=ROOT/path
 image=PILImage.open(file).convert('RGB')
 # Preserve all original annotations while removing only blank outer padding.
 from PIL import ImageChops
 box=ImageChops.difference(image,PILImage.new('RGB',image.size,'white')).point(lambda x:255 if x>18 else 0).getbbox()
 if box:
  x0,y0,x1,y1=box;pad=18;image=image.crop((max(0,x0-pad),max(0,y0-pad),min(image.width,x1+pad),min(image.height,y1+pad)))
 cropped=ASSETS/('placed-'+file.name);image.save(cropped)
 widths={'car-t-dynamics.png':416,'project-workflow.png':440,'shared-cell-states.png':450,'ot-candidate-paths.png':460,'token-embedding.png':425,'encoder-block.png':468,'representative-trajectory.png':468,'structure-objective.png':305,'coverage-loss.png':305,'complexity-loss.png':305}
 width=widths[file.name];height=width*image.height/image.width
 story.append(Image(str(cropped),width,height))

def figure(kind):
 # Re-render the exact overview topology and class colors from docs/transformer.md.
 src=(ROOT/'docs/transformer.md').read_text(encoding='utf-8')
 mermaid=src.split('```mermaid',1)[1].split('```',1)[0].strip()
 (ASSETS/'architecture.mmd').write_text(mermaid+'\n')
 labels=dict(re.findall(r'(\w+)\["([^"\n]+)"\]',mermaid))
 d=Drawing(468,332)
 groups={'encoder':('#dbeafe','#2563eb','#172554'),'processing':('#fef3c7','#b45309','#451a03'),'decoder':('#dcfce7','#15803d','#052e16')}
 def box(id,x,y,w,h,lines,group):
  fill,stroke,ink=map(colors.HexColor,groups[group])
  d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=fill,strokeColor=stroke,strokeWidth=.85))
  for n,t in enumerate(lines):d.add(String(x+w/2,y+h/2+(len(lines)-1)*6-n*12-3,t,fontName='Sans',fontSize=9,textAnchor='middle',fillColor=ink))
 def arrow(x,y,xx,yy,color='#475569',dash=False):
  col=colors.HexColor(color)
  line=Line(x,y,xx,yy,strokeColor=col,strokeWidth=.9)
  if dash:line.strokeDashArray=[4,3]
  d.add(line);a=math.atan2(yy-y,xx-x)
  d.add(Polygon([xx,yy,xx-5*math.cos(a)+2*math.sin(a),yy-5*math.sin(a)-2*math.cos(a),xx-5*math.cos(a)-2*math.sin(a),yy-5*math.sin(a)+2*math.cos(a)],fillColor=col,strokeColor=col))
 x,w=151,310
 chain=[('I',300,26,['Input: patient paths and shared state IDs'],'encoder'),('EM',264,26,[labels['EM']],'encoder'),('E1',228,26,[labels['E1']],'encoder'),('E2',192,26,[labels['E2']],'encoder'),('EO',156,26,['Encoder output: contextual path tokens'],'processing'),('P',110,36,['Processing: mean pooling +','mass-aware aggregation'],'processing'),('D',64,28,[labels['D']],'decoder')]
 for id,y,h,lines,g in chain:box(id,x,y,w,h,lines,g)
 for k in range(len(chain)-1):arrow(x+w/2,chain[k][1],x+w/2,chain[k+1][1]+chain[k+1][2])
 box('EQ',2,64,132,28,['Temporal-edge queries'],'processing');arrow(134,78,151,78)
 box('H1',151,8,130,32,['Head 1: predict','OT matrices'],'decoder')
 box('H2',289,8,173,32,['Head 2: representative','backbone and branches'],'decoder')
 arrow(275,64,210,40);arrow(337,64,375,40)
 # Dotted OT-loss feedback follows the overview, to Encoder block 1.
 arrow(151,24,143,24,'#2563eb',True)
 d.add(Line(143,24,143,241,strokeColor=colors.HexColor('#2563eb'),strokeWidth=.9,strokeDashArray=[4,3]))
 arrow(143,241,151,241,'#2563eb',True)
 txt=String(0,0,'Stage 1: OT-loss backpropagation',fontName='Sans',fontSize=8,fillColor=colors.HexColor('#2563eb'))
 label=Group(txt);label.translate(130,124);label.rotate(90);d.add(label)
 cropped=d
 renderPDF.drawToFile(cropped,str(ASSETS/'architecture.pdf'));story.append(cropped)

INLINE=ASSETS/'inline';INLINE.mkdir(exist_ok=True)
inline_parser=MathTextParser('agg')
inline_cache={}
def inline_math(expr,size=10.5):
 key=(expr,size)
 if key not in inline_cache:
  parsed=inline_parser.parse('$'+expr+'$',dpi=600,prop=FontProperties(size=size))
  alpha=np.asarray(parsed.image)
  rgba=np.zeros((*alpha.shape,4),dtype=np.uint8);rgba[:,:,3]=alpha
  file=INLINE/(hashlib.sha256((expr+str(size)).encode()).hexdigest()[:20]+'.png');PILImage.fromarray(rgba).save(file)
  h,w=alpha.shape;scale=72/600
  inline_cache[key]=f'<img src="{file}" width="{w*scale:.4f}" height="{h*scale:.4f}" valign="{-parsed.depth*scale:.4f}"/>'
 return inline_cache[key]
def rich(text,math_size=10.5):
 chunks=re.split(r'(\$[^$]+\$|`[^`]+`)',text.replace('\n',' '))
 result=[]
 for chunk in chunks:
  if chunk.startswith('$') and chunk.endswith('$'):result.append(inline_math(chunk[1:-1],math_size))
  elif chunk.startswith('`') and chunk.endswith('`'):result.append('<font name="Courier" size="8.7">'+escape(chunk[1:-1])+'</font>')
  else:
   chunk=escape(chunk);chunk=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',chunk);result.append(chunk)
 return ''.join(result)

def maketable(lines):
 data=[[Paragraph(rich(x.strip(),8),styles['cell']) for x in row.split('|')] for row in lines]
 n=len(data[0]);t=Table(data,colWidths=[468/n]*n,repeatRows=1,hAlign='CENTER')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('LINEABOVE',(0,0),(-1,0),.7,colors.HexColor('#20384d')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#20384d')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#20384d')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
 items=[t]
 if story and isinstance(story[-1],Paragraph) and story[-1].style.name in ('h1','h2'):
  items.insert(0,story.pop())
 story.extend([KeepTogether(items),Spacer(1,9)])

text=(HERE/'manuscript.md').read_text(encoding='utf-8-sig')
blocks=text.strip().split('\n\n')
for b in blocks:
 b=b.strip()
 if b.startswith('$$'):equation(b[2:-2].strip())
 elif b.startswith('@figure '):figure(b.split()[1])
 elif b.startswith('@image '):existing_image(b[7:].strip())
 elif b=='@page':story.append(PageBreak())
 elif b.startswith('@table\n'):maketable(b.splitlines()[1:])
 elif b.startswith('# '):story.append(Paragraph(escape(b[2:]),styles['title']))
 elif b.startswith('### '):story.append(Paragraph(escape(b[4:]),styles['h2']))
 elif b.startswith('## '):story.append(Paragraph(escape(b[3:]),styles['h1']))
 elif b.startswith('@meta '):story.append(Paragraph(escape(b[6:]),styles['sub']))
 elif b.startswith('@caption '):
  caption=Paragraph(escape(b[9:]),styles['caption'])
  if story and isinstance(story[-1],(Image,Drawing)):
   picture=story.pop();story.append(KeepTogether([picture,Spacer(1,5),caption]))
  else:story.append(caption)
 else:
  story.append(Paragraph(rich(b),styles['p']))
def footer(c,doc):
 c.saveState();c.setStrokeColor(colors.HexColor('#b5bec8'));c.setLineWidth(.4);c.line(72,43,540,43);c.setFont('Sans',7);c.setFillColor(colors.HexColor('#586675'));c.drawRightString(540,31,str(doc.page));c.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=(612,792),leftMargin=72,rightMargin=72,topMargin=48,bottomMargin=57,title='OT-Transformer Encoder-Decoder for Representative Patient Trajectories with Important Branches',author='JING-WEI CHEN',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT);print('Equations:',eqcount)








from PIL import Image
from pathlib import Path
import numpy as np,json,csv
root=Path(__file__).parent;metrics={};points=[]
colors=[(237,177,32),(19,113,190),(221,84,0),(58,170,50),(133,21,209)]
for name,idx,x0,x4,yzero,scale in [('bs1',25,85,1513,819.25,50000/110.5),('glofitamab',28,82,1552,922,10000/78.5)]:
 im=np.array(Image.open(root/'reference_images'/f'original_{idx}.png').convert('RGB')).astype(float)
 data=np.genfromtxt(root/(name+'_timecourse.csv'),delimiter=',',skip_header=1)
 n=3 if name=='bs1' else 5;result=[]
 palette=[colors[0],colors[1],colors[2]] if name=='bs1' else colors
 for j,c in enumerate(palette):
  mask=np.linalg.norm(im-np.array(c),axis=2)<25
  # Exclude source legend, ticks, titles and edges; inspect plotted data only.
  mask[:45]=False;mask[:, :x0+5]=False;mask[:,x4-5:]=False
  if name=='bs1':mask[150:280,1220:1495]=False;mask[824:]=False
  else:mask[48:242,935:1543]=False;mask[930:]=False
  measured=[]
  for x in range(x0+15,x4-10,7):
   yy=np.where(mask[:,x])[0]
   if not len(yy):continue
   t=4*(x-x0)/(x4-x0);v=(yzero-float(np.median(yy)))*scale
   predicted=np.interp(t*3600,data[:,0],data[:,j+1]);measured.append([t,v,predicted]);points.append([name,j,t,v,predicted])
  a=np.array(measured);err=a[:,1]-a[:,2]
  result.append({'series':j,'sample_count':len(a),'rmse_complexes_per_cell':float(np.sqrt(np.mean(err**2))),'rmse_pixels':float(np.sqrt(np.mean(err**2))/scale),'max_absolute_error_complexes_per_cell':float(max(abs(err)))})
 metrics[name]=result
(root/'screenshot_comparison.json').write_text(json.dumps(metrics,indent=2))
with (root/'screenshot_samples.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','series','hours','image_value','model_value']);w.writerows(points)
print(json.dumps(metrics,indent=2))

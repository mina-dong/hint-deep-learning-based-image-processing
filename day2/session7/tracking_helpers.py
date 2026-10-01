from pathlib import Path
import cv2
import numpy as np

def candidates(frame, color='green'):
    low, high = ((35,80,60),(85,255,255)) if color == 'green' else ((17,135,140),(40,255,255))
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), low, high)
    found = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(contour)
        if area < 80:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        found.append({'xy':(x+w/2,y+h/2), 'box':(x,y,w,h), 'area':area})
    return sorted(found, key=lambda c:c['xy'][0]), mask

def choose(found, previous, point_distance, within_limit, max_distance):
    if previous is None:
        return (0,'start',None) if len(found)==1 else (None,'wait',None)
    if not found:
        return None,'missing',None
    distances = [point_distance(previous,c['xy']) for c in found]
    best = int(np.argmin(distances))
    if sum(abs(d-distances[best]) < 1e-9 for d in distances) > 1:
        return None,'tie',distances[best]
    if not within_limit(distances[best],max_distance):
        return None,'too_far',distances[best]
    return best,'linked',distances[best]

def overlay(frame, found, chosen=None, trail=(), previous=None, distance_lines=False, manual=False):
    result = frame.copy()
    for i,c in enumerate(found):
        x,y,w,h = c['box']
        cv2.rectangle(result,(x,y),(x+w-1,y+h-1),(0,220,255),2)
        q=tuple(int(round(v)) for v in c['xy'])
        cv2.circle(result,q,4,(0,220,255),-1)
        cv2.putText(result,str(i+1),(x,max(17,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2)
        if previous is not None and distance_lines:
            p=tuple(int(round(v)) for v in previous)
            cv2.line(result,p,q,(255,220,0),2)
            d=np.hypot(c['xy'][0]-previous[0],c['xy'][1]-previous[1])
            cv2.putText(result,f'{d:.1f}',((p[0]+q[0])//2,(p[1]+q[1])//2),0,.5,(255,255,255),2)
    if len(trail)>1:
        cv2.polylines(result,[np.round(trail).astype(np.int32)],False,(255,100,20),3)
    if previous is not None:
        p=tuple(int(round(v)) for v in previous)
        cv2.drawMarker(result,p,(255,220,0),cv2.MARKER_CROSS,16,2)
        cv2.putText(result,'P',(p[0]+6,p[1]-6),0,.6,(255,220,0),2)
    if chosen is not None:
        q=tuple(int(round(v)) for v in found[chosen]['xy'])
        if manual:
            cv2.drawMarker(result,q,(17,90,197),cv2.MARKER_DIAMOND,20,3)
        else:
            cv2.circle(result,q,10,(0,0,255),3)
    return result

def read_image(path):
    a=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
    if a is None: raise FileNotFoundError(path)
    return a

def show(images,titles):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,len(images),figsize=(5*len(images),4),squeeze=False)
    for ax,im,t in zip(axs[0],images,titles):
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB) if im.ndim==3 else im,cmap='gray',vmin=0,vmax=255)
        ax.set_title(t);ax.axis('off')
    plt.tight_layout();plt.show();plt.close(fig)

def track_video(path,point_distance,within_limit,max_distance=35,color='green',output='outputs/tracked.avi'):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError('Cannot open input')
    writer=None;previous=None;trail=[];records=[];segment=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            found,_=candidates(frame,color)
            chosen,status,d=choose(found,previous,point_distance,within_limit,max_distance)
            if chosen is None:
                previous=None;trail=[]
            else:
                if status=='start':segment+=1
                previous=found[chosen]['xy'];trail.append(previous);trail=trail[-30:]
            result=overlay(frame,found,chosen,trail)
            cv2.putText(result,f'{len(records)}: {status} / segment {segment}',(10,25),0,.55,(255,255,255),2)
            if writer is None:
                Path(output).parent.mkdir(parents=True,exist_ok=True)
                fps=cap.get(cv2.CAP_PROP_FPS) or 30
                writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'FFV1'),fps,(frame.shape[1],frame.shape[0]))
                if not writer.isOpened():raise RuntimeError('Cannot open output')
            writer.write(result)
            records.append({'frame':len(records),'status':status,'segment':segment,'count':len(found),'chosen':chosen,'xy':previous,'distance':d})
        if records:
            cv2.imencode('.png',result)[1].tofile(str(Path(output).with_suffix('.png')))
    finally:
        cap.release()
        if writer is not None:writer.release()
    return records

// Gyro Archery V2.1 — optional isolated inference worker.
import {GestureRecognizer, FilesetResolver} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/+esm';
const BASE='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm';
const MODEL='https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';
let engine;
self.onmessage=async e=>{
 const m=e.data;
 if(m.type==='init'){
   try {
    engine=await GestureRecognizer.createFromOptions(await FilesetResolver.forVisionTasks(BASE),{
     baseOptions:{modelAssetPath:MODEL},runningMode:'VIDEO',numHands:2,
     minHandDetectionConfidence:.48,minHandPresenceConfidence:.48,minTrackingConfidence:.45
    });
    self.postMessage({type:'ready'});
   }catch(err){self.postMessage({type:'error',error:String(err)});}
   return;
 }
 if(m.type==='frame'){
   const t=performance.now();
   try{
    const result=engine.recognizeForVideo(m.bitmap,m.ts);
    const hands=(result.landmarks||[]).map((lm,i)=>{
      let found=(result.gestures?.[i]||[]).filter(c=>c.categoryName==='Closed_Fist'||c.categoryName==='Open_Palm').sort((a,b)=>b.score-a.score)[0];
      let x=0,y=0;
      for(const p of lm){x+=p.x;y+=p.y;}
      return {x:x/lm.length,y:y/lm.length,label:found?.categoryName||'UNKNOWN',confidence:found?.score||0,handedness:result.handedness?.[i]?.[0]?.categoryName||'?'};
    });
    self.postMessage({type:'result',seq:m.seq,ts:m.ts,elapsed:performance.now()-t,hands});
   }catch(err){self.postMessage({type:'frameError',seq:m.seq,error:String(err)});}
   finally{m.bitmap.close();}
 }
};

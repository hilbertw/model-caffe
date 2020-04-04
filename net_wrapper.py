class dup_blob_rec:
  def __init__(self,_origin_blob_idx,_clone_blob_idx):
      self.origin_blob_idx=_origin_blob_idx
      self.clone_blob_idx=_clone_blob_idx

class net_wrapper:
  
  def __init__(self,_net):
      self.net=_net
      self.records=[]
      self.top_ids=[]
      self.bottom_ids=[]
      self._blobs=[]
      self._blob_names=[]
      self.build()

  def add_blob(self,blob_idx):
      new_idx = len(self._blobs)
      self._blob_names.append('%s_clone'%(self.net._blob_names[blob_idx]))
      self._blobs.append(self.net._blobs[blob_idx])
      r=dup_blob_rec(blob_idx,new_idx)
      self.records.append(r)
      return new_idx

  def convert_blob(self,blob_idx):
       for r in self.records:
           if r.origin_blob_idx==blob_idx :
#                print("convert %d to %d"%(blob_idx, r.clone_blob_idx))
                return r.clone_blob_idx
       return blob_idx 

  def count(self):
      return len(self._blobs)

  def _top_ids(self,i):
      return self.top_ids[i]

  def _bottom_ids(self,i):
      return self.bottom_ids[i]

  def convert(self,s):
      r=[]
      for b in s:
          n=self.convert_blob(b)
          r.append(n)
      
      return r

  def build(self):
     for n in self.net._blob_names:
        self._blob_names.append(n)
 
     for b in self.net._blobs:
        self._blobs.append(b)

     i=0
     for l in self.net.layers:
        top=self.net._top_ids(i)
        bottom=self.net._bottom_ids(i)
#        if len(top)>0 and len(bottom)>0:
#           print("%d:%d,%d"%(i,top[0],bottom[0]))
        if len(top)==1 and len(bottom)==1 and top[0]==bottom[0]:
           idx=top[0]
           new_idx=self.add_blob(idx)
           self.top_ids.append([new_idx]) 
           self.bottom_ids.append([idx]) 
        else:
           _top=self.convert(top)
           _bottom=self.convert(bottom)
           self.top_ids.append(_top)
           self.bottom_ids.append(_bottom)
        i=i+1

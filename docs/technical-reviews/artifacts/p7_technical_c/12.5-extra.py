x=torch.tensor([1.,0.,-1.,0.]); y=torch.roll(x,1); a=x.clone(); b=y.clone(); c=torch.tensor([1.,-1.,1.,-1.])
print('four-point',[(float(z@a),float(z@b),float((z@a)**2+(z@b)**2),float(z@c)) for z in [x,y]])
print('complex3+4j',abs(3+4j),abs(3+4j)**2)
for f in [440.,480.,450.]:
 s=torch.stft(tone(f,seconds=.2),400,160,window=torch.hann_window(400),return_complex=True);p=s.abs().square().mean(-1)
 print('frequency-control',f,int(p.argmax()),float(p[11]),float(p[12]))
s=torch.stft(tone(440.,seconds=.2),400,160,window=torch.hann_window(400),return_complex=True,center=False)
print('center_false',tuple(s.shape))
assert tuple(s.shape)==(201,18)

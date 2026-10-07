import math
for speed in [60,15]:
 print('speed_deg_per_grid',speed,'period',360/speed)
 for scale in [1,2,4]:
  angle=speed/scale;print('scale',scale,'neighbor_angle',angle,'unit_dot',math.cos(math.radians(angle)))

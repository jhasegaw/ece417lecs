for x in range(0,M):
  for y in range(0,N):
    (u,v) = input_pixels_corresponding_to(x,y)
    J[y,x] = compute_pixel(I,v,u)      

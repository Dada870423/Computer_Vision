def subsampling(img, sub_size):
    size_w = img.shape[0]
    size_h = img.shape[1]
    new_w = math.floor(size_w / sub_size)
    new_h = math.floor(size_h / sub_size)
    new = np.zeros((new_w, new_h))
    #print(size_w,size_h)
    #print(new_w,new_h)
    for x in range(0, new_w):
        for y in range(0, new_h):
            new[x, y] = img[x*sub_size, y*sub_size]
            #print("x y :", x*sub_size, y*sub_size)
    return new
    

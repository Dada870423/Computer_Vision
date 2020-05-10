class RANSAC():
    def __init__(self, thresh, n_times, points):
        self.thresh = thresh
        self.n_times = n_times
        self.points = points
	
    def calH(RanPoints):
    	return H


    def ransac(self, CorList):
        print("hihi", self.n_times)
        MaxLines = []
        H = None
        Clen = len(CorList)
        for i in range(self.n_times):
        	## pick up 4 random points
            Cor1 = CorList[random.randrange(0, Clen)]
            Cor2 = CorList[random.randrange(0, Clen)]
            Cor3 = CorList[random.randrange(0, Clen)]
            Cor4 = CorList[random.randrange(0, Clen)]
            RanPoints = np.vstack((Cor1, Cor2, Cor3, Cor4))
            ## cal H
            H = calH(RanPoints = RanPoints)
            ## Cal line
            Lines = []


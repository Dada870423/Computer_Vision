class RANSAC():
    def __init__(self, thresh, n_times, points):
        self.thresh = thresh
        self.n_times = n_times
        self.points = points
	
    def ransac(self):
        print("hihi", self.n_times)

"""
RSC = RANSAC(thresh = 10.0, n_times = 100, points = 4)
RSC.ransac()
"""


(1) hstack for data is at the bottom (last diemnsion), run fixed-gmm and l2 again
(2) EM version -- (2 * self.b + np.sum(responsibility * np.square(self.w[:-1]), axis=0))

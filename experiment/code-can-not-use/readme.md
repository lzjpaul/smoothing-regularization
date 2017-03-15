(1) bias is not at the last dimension (vstack, hstack)
(2) gmm-em version: (2 * self.b + np.sum(responsibility * np.square(self.w[:-1]), axis=0)) not reg_lambda * np.square(**)

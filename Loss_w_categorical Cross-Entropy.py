import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [ 1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] + 
		 math.log(softmax_output[1])*target_output[1] + 
		 math.log(softmax_output[2])*target_output[2] )

print(loss)

#simplified

loss = -(math.log(softmax_output[target_output.index(1)]))

print(loss)

#using numpy

import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
							[0.1, 0.5, 0.4],
							[0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

print(-np.log(softmax_outputs[[0, 1, 2], class_targets]))
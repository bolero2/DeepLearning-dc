import numpy as np
import matplotlib,pylab as plt
import scipy.io

matfile_name = "C:\\dataset\\mpii_human_pose_v1_u12_2\\mpii_human_pose_v1_u12_1.mat"
matfile = scipy.io.loadmat(matfile_name, struct_as_record=False)
print(matfile.get('RELEASE')[0, 0].get('annolist'))
# print(matfile['re'])
# # print(type(matfile))

# for i in matfile:
#     print(i)

# print(type(matfile['RELEASE'])) # <class 'numpy.ndarray'>
temp = matfile['RELEASE']
temp = temp[0, 0]
print(type(temp))
print(temp.__dict__['annolist'][0, 0].__dict__['annorect'][0, 0].__dict__['annopoints.point'][0, 0].__dict__['point'])
# temp = np.squeeze(temp)
# print(temp)
exit(0)
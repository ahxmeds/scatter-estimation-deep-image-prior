#%%
import numpy as np 
import matplotlib.pyplot as plt 


# %%
mean = 200
std = 4
num_samples = 100
mean_list = np.random.normal(200, 3, 100)
s1, s2 = [], []
for i in range(100):
    ss1 = np.random.normal(mean_list[i], std, 1)[0]
    ss2 = np.random.normal(mean_list[i], std, 1)[0]
    s1.append(ss1)
    s2.append(ss2)

#%%
s1 = np.array(s1)
s2 = np.array(s2)
x = (s1 + s2)/2
y = (s2 - s1)/x
z = np.log(s2) - np.log(s1)

fig, ax = plt.subplots()
ax.plot(x, y, 'o')
ax.plot(x, z, 'x')

# %%

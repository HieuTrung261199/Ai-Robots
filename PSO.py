import numpy as np
import matplotlib.pyplot as plt
import math

# Buoc 1: Khoi tao
n = 100            # 100 ca the
npar = 3           # 3 tham so
Min = 0
Max = 1
c1 = 1
c2 = 3
max_iteration = 100

def objective_function(X):
    # Đây là một hàm mục tiêu giả định, bạn cần thay thế bằng hàm mục tiêu thực tế của mình
    A = 10
    return A*2 + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])

# Tạo ma trận có giá trị từ Min đến Max
p = Min + (Max - Min) * np.random.rand(n, npar)

# Tạo ma trận v co kich thuoc bang voi ma tran p
v = np.zeros((n, npar))

# Tạo ma trận sai so j
j = np.zeros((n, 1)) # co 1 sai so j cho moi ca the

# Lưu ma trận vào file txt
np.savetxt('random_p.txt', p)
np.savetxt('random_v.txt', v)

gbest = float('inf')  # Khởi tạo giá trị tốt nhất cục bộ là vô cực
pbest = np.zeros((n, npar))  # Ma trận vị trí tốt nhất cục bộ của từng hạt

# Lưu kết quả tốt nhất của từng lần lặp
gbest_history = []

# Buoc 2: Danh gia
iteration = 0

while iteration < max_iteration:
    for i in range(n):
        fitness = objective_function(p[i])
        if fitness < j[i]:
            j[i] = fitness
            pbest[i] = p[i]
            
    gbest_index = np.argmin(j)
    gbest = j[gbest_index]
    gbest_history.append(gbest)
    
    # Cập nhật vận tốc và vị trí của mỗi hạt
    for i in range(n):
        v[i] = c1 * np.random.rand() * (pbest[i] - p[i]) + c2 * np.random.rand() * (gbest - p[i])
        p[i] = p[i] + v[i]
    
    iteration += 1
    print("Iteration: {}, Best Fitness: {}".format(iteration, gbest))

print("Global Best Position:", p[gbest_index])

# Vẽ biểu đồ minh họa
plt.plot(gbest_history)
plt.xlabel('Iterations')
plt.ylabel('Best Fitness Value')
plt.title('PSO Optimization')
plt.grid(True)
plt.show()

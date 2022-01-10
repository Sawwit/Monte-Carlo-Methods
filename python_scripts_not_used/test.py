import numpy as np

y = np.arange(0,4**4)

y = y.reshape((4,4,4,4))
# print(y)

def potential_v(x,lamb):
    '''Compute the potential function V(x).'''
    return lamb*(x*x-1)*(x*x-1)+x*x

def neighbor_sum(phi,s):
    '''Compute the sum of the state phi on all 8 neighbors of the site s.'''
    w = len(phi)

    return (phi[(s[0]+1)%w,s[1],s[2],s[3]] + phi[(s[0]-1)%w,s[1],s[2],s[3]] +
            phi[s[0],(s[1]+1)%w,s[2],s[3]] + phi[s[0],(s[1]-1)%w,s[2],s[3]] +
            phi[s[0],s[1],(s[2]+1)%w,s[3]] + phi[s[0],s[1],(s[2]-1)%w,s[3]] +
            phi[s[0],s[1],s[2],(s[3]+1)%w] + phi[s[0],s[1],s[2],(s[3]-1)%w] )

# def neighbor_sum(phi,s):
#     '''Compute the sum of the state phi on all 8 neighbors of the site s.'''
#     w = len(phi)

#     return (phi[(s[0]+1)%w,s[1]] + phi[(s[0]-1)%w,s[1]] +
#             phi[s[0],(s[1]+1)%w] + phi[s[0],(s[1]-1)%w] )

def naive_check_potential(y):
    y_check = np.zeros_like(y)
    for i in range(len(y.shape)):
        for j in range(len(y.shape)):
            for k in range(len(y.shape)):
                for l in range(len(y.shape)):
                    y_check[i,j,k,l] = potential_v(y[i,j,k,l], 1)
    return y_check

def naive_check(y):
    y_check = np.zeros_like(y)
    for i in range(len(y.shape)):
        for j in range(len(y.shape)):
            for k in range(len(y.shape)):
                for l in range(len(y.shape)):
                    y_check[i,j,k,l] = neighbor_sum(y,[i,j,k,l])
    return y_check

def naive_check_sum(y):
    y_check = 0
    for i in range(len(y.shape)):
        for j in range(len(y.shape)):
            for k in range(len(y.shape)):
                for l in range(len(y.shape)):
                    y_check += y[i,j,k,l]
    return y_check

def naive_check_force(y, kappa, lamb):
    width = len(y.shape)
    F= np.zeros((width,width,width,width))
    for i in range(width):
        for j in range(width):
            for k in range(width):
                for l in range(width):
                    F[i,j,k,l]= 2*y[i,j,k,l] + 4*lamb*(y[i,j,k,l]**2-1)*y[i,j,k,l]-2*kappa*neighbor_sum(y,[i,j,k,l])

    return F 




# y_check = np.zeros((2,2))
# for i in range(2):
#     for j in range(2):
#         y_check[i,j] = neighbor_sum(y,[i,j])
def potential_matrix(phi):
    return potential_v(phi,1)

def neighbor_sum_matrix(phi):
    A = np.zeros_like(phi)
    A += (np.roll(phi, 1, axis = 0) + np.roll(phi, -1, axis = 0) + np.roll(phi, 1, axis = 1) + np.roll(phi, -1, axis = 1) 
         + np.roll(phi, 1, axis = 2) + np.roll(phi, -1, axis = 2) + np.roll(phi, 1, axis = 3) + np.roll(phi, -1, axis = 3))
    return A

def force(phi,kappa,lamb):
    width = len(phi.shape)
    F = np.zeros((width,width,width,width))
    F += 2*phi + 4*lamb*(phi**3) - 4*lamb*phi - 2 * kappa * (np.roll(phi, 1, axis = 0) + np.roll(phi, -1, axis = 0) + np.roll(phi, 1, axis = 1) + np.roll(phi, -1, axis = 1) 
         + np.roll(phi, 1, axis = 2) + np.roll(phi, -1, axis = 2) + np.roll(phi, 1, axis = 3) + np.roll(phi, -1, axis = 3))
    return F

# y_test = neighbor_sum_matrix(y)

# print("y_test", y_test)
# print("y_check", y_check)
print(y)
print(len(y.shape))
print("check: \n", naive_check_force(y,1,1))
print("test: \n", force(y,1,1))
print("difference: \n", (naive_check_force(y,1,1) - force(y,1,1)))


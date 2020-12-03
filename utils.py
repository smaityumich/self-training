import numpy as np
import matplotlib.pyplot as plt


def generate_x_y_a(p_mx, n = 100, mu_mult = 1, cov_mult = 0.5):
    data_y = []
    data_a = []
    data_x = []
    for y in [0, 1]:
        for a in [0, 1]:
            n_ya = int(n*p_mx[y][a])
            data_y += n_ya*[y]
            data_a += n_ya*[a]
            data_x.append(np.random.normal(loc=np.array([y, a])*mu_mult, scale=cov_mult, size=(n_ya,2)))

    order = np.random.permutation(len(data_y))
    data_x = np.vstack(data_x)[order]
    data_y = np.array(data_y)[order]
    data_a = np.array(data_a)[order]
    
    return data_x, data_a, data_y
    

def sim_x_y(prop, n = 100, mu = 1, sigma = 1):
    data_y = np.random.binomial(1, prop, (n,1))
    data_x = np.random.normal(scale=sigma, size=(n, 2)) + data_y @ np.array([[mu]*2])
    return data_x, data_y.reshape((-1,))

def plot_x_y(data_x, data_y, line = None, ax = None, y_ticks = True):
    colors = ['r','b']
    markers = ['x', 'o']
    
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
    for y in [0,1]:
        x_y = data_x[data_y==y]
        ax.scatter(x_y[:,0],x_y[:,1], c=colors[y], marker = markers[y], s=75, label='y=%d' % (y,))
    if line is not None:
        ax.plot(line[0], line[1], 'k-')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid()
    
    if not y_ticks:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    
    return ax



def simul_x_y_a(prop_mtx, n=100, mu_mult=1., cov_mult=0.5, skew=2.):
    
    mu_y0_a0 = np.array([1.,1.])*mu_mult
    mu_y0_a1 = np.array([5., 4.])*mu_mult
    mu_y1_a0 = np.array([1.,3.])*mu_mult
    mu_y1_a1 = np.array([3., 4.])*mu_mult
    
    mu = [[mu_y0_a0, mu_y0_a1], [mu_y1_a0, mu_y1_a1]]
    
    cov_y0_a0 = np.array([skew,1.])*cov_mult
    cov_y0_a1 = np.array([1.,skew])*cov_mult
    cov_y1_a0 = np.array([skew,1.])*cov_mult
    cov_y1_a1 = np.array([1.,skew])*cov_mult
    
    cov = [[cov_y0_a0, cov_y0_a1], [cov_y1_a0, cov_y1_a1]]
    
    data_x = []
    data_y = []
    data_a = []
    
    for y in [0,1]:
        for a in [0,1]:
            n_ya = int(n*prop_mtx[y][a])
            data_y += n_ya*[y]
            data_a += n_ya*[a]
            data_x.append(np.random.normal(loc=mu[y][a], scale=np.sqrt(cov[y][a]), size=(n_ya,2)))
    
    order = np.random.permutation(len(data_y))
    data_x = np.vstack(data_x)[order]
    data_y = np.array(data_y)[order]
    data_a = np.array(data_a)[order]
    
    return data_x, data_a, data_y
    
def reductions_prob(mitigator, x, n_sample=20):
    prob = mitigator.predict(x)
    for _ in range(n_sample-1):
        prob += mitigator.predict(x)
    return prob/n_sample

def sample_balanced(y, y_protected, prop=0.9):

    idx = []
    for c_y in [0,1]:
        if c_y == 0:
            protected_c_y = y_protected[y == c_y]
            levels, counts = np.unique(protected_c_y, return_counts=True)
            n_max = min(counts)
            n = int(prop*n_max)
        else:
            n = int(n*y.sum()/(1-y).sum())
        for c in levels:
            idx_c = np.where(np.logical_and(y == c_y, y_protected==c))[0]
            if n <= len(idx_c):
                sample_idx = np.random.choice(idx_c, size=n, replace=False)
            else:
                sample_idx = np.random.choice(idx_c, size=n, replace=True)
                
            idx += sample_idx.tolist()
        
    return np.array(idx)

def plot_sample(data_x, data_a, data_y, ax=None, y_ticks=True):
    markers = ['o' , 'x']
    colors = ['r','b']
    
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
    for y in [0,1]:
        for a in [0,1]:
            x_ya = data_x[np.logical_and(data_a==a, data_y==y)]
            ax.scatter(x_ya[:,0],x_ya[:,1], c=colors[y], marker=markers[a], s=75, label='a=%d, y=%d' % (a,y))
    ax.legend(loc='upper left', fontsize=20)
    ax.grid()
    
    if not y_ticks:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    
    return ax


def plot_scatter_sample(data_x, data_y, data_a, line = None, line_spec = [0, 0], ax=None, y_ticks=True):
    colors = ['r','b']
    markers = ['x', 'o']
    
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
    for y in [0,1]:
        for a in [0,1]:
            x_ya = data_x[np.logical_and(data_y==y, data_a== a)]
            ax.scatter(x_ya[:,0],x_ya[:,1], c=colors[y], marker=markers[a], s=75, label='y=%d, a=%d' % (y,a))
    if line is not None:
        ax.plot(line[0], line[1], 'k--')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid()
    
    if not y_ticks:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    
    return ax

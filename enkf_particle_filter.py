import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sn

def plot_speeds(X_true, Y_obs, x_a_enkf):
    ### plot speed (true, observed, filtered)
    plt.figure(1)
    plt.plot(X_true[2, :], X_true[3, :], color='b', label='True state')
    plt.plot(Y_obs[0, :], Y_obs[1, :], color='g', label='Observations')
    plt.plot(x_a_enkf[2, :], x_a_enkf[3, :], color='r', label='EnKF')
    plt.xlabel("dx/dt")
    plt.xlabel("dy/dt")
    plt.legend()
    plt.show()

def plot_composantes(X_true, Y_obs, x_a_enkf, P_a_enkf,nb):
    ### plot trajectories (true and filtered)
    names=['x','y','vx','vy']
    time = np.arange(0, nb,1)
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(X_true[i, :])


        plt.plot(time,x_a_enkf[i, :], 'r')
        if i==2:
            plt.plot(time,Y_obs[0, :], 'k*')
            plt.legend(['truth',  'EnKF','obs'], prop={'size': 20}, loc=1)
        elif i==3:
            plt.plot(time,Y_obs[1, :], 'k*')
            plt.legend(['truth','EnKF','obs'], prop={'size': 20}, loc=1)
        else :
            plt.legend(['truth', 'EnKF'], prop={'size': 20}, loc=1)

        plt.fill_between(time,x_a_enkf[i, :] - 1.96 * sqrt(P_a_enkf[i, i, :]),
                     x_a_enkf[i, :] + 1.96 * sqrt(P_a_enkf[i, i, :]), facecolor='red', alpha=0.5)
        plt.ylabel(names[i], size=20)

    plt.show()

def plot_bathy_time(Y_obs, x_a_enkf, P_a_enkf,nb):
    bathy_a=g(x_a_enkf[0,:],x_a_enkf[1,:])
    time = np.arange(0, nb, 1)

    plt.plot(time, Y_obs[2, :], 'k*')
    plt.plot(time, bathy_a, 'r')
    plt.legend(['obs','EnKF'], prop={'size': 20}, loc=1)

    plt.show()

def plot_traj(X_true,x_a_enkf):
    ### plot trajectories (true,  filtered)
    plt.figure(2)
    plt.scatter(x_a_enkf[0, 0], x_a_enkf[1, 0],color='black')
    plt.plot(X_true[0, :], X_true[1, :], 'b', label='True state')
    plt.plot(x_a_enkf[0, :], x_a_enkf[1, :], 'r', label='EnKF')
    x_grd, y_grd = np.meshgrid(np.linspace(-12,12,100), np.linspace(-12,12,100))
    plt.contourf(x_grd, y_grd, g(x_grd,y_grd))
    plt.xlabel("x")
    plt.xlabel("y")
    plt.legend()
    plt.show()

def plot_density(particles,truth):
    n,t = np.shape(truth)
    plt.figure()
    for i in range(0,t,10):
        mylist = particles[0, :,i].tolist()
        mylist.append(X_true[0, i])
        sn.kdeplot(np.array(mylist))
    plt.xlabel('x')
    plt.title("Reoartition of x for each 10 step")
    plt.show()

def psi(x,y,L):
    return L*(np.cos(0.5*x) + np.cos(0.5*y))

def generate_obs(init_x, init_y, init_vx, init_vy, p, n, nb, dT, R, t0=0):
    ''' Generate observation and true state
    Y the observations
    X_true true value of state (buoy position and speed)'''
    Y = np.zeros((p, nb))
    X_true = np.zeros((n, nb))
    Y[:, 0] = np.array([init_vx, init_vy, g(init_x, init_y)]).T
    x_past = np.array([init_x, init_y, init_vx, init_vy])
    X_true[:, 0] = x_past.T

    for t in range(1, nb):
        # Runge-Kutta (4,5) integration method
        X1 = np.copy(x_past)
        k1 = f(X1)
        X2 = np.copy(x_past + k1 / 2 * dT)
        k2 = f(X2)
        X3 = np.copy(x_past + k2 / 2 * dT)
        k3 = f(X3)
        X4 = np.copy(x_past + k3 * dT)
        k4 = f(X4)

        # return the state in the near future
        x_past = x_past + dT / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        X_true[:, t] = x_past.T
        Y[:, t] = [x_past[2], x_past[3], g(x_past[0], x_past[1])] + random.multivariate_normal(zeros(p), R)

    return Y, X_true


def g(x, y):
    ''' Bathymetry model'''
    goutte =-30 + 20 * (1 - np.exp(-0.01 * (x ** 2 + y ** 2)))
    pente = - 30 + 0.1 * (x+y)
    plan = -30
    return goutte


def h_obs(x_past):
    Y = np.zeros((3, 1))
    Y = np.array([0,0,0])
    Y[0] = x_past[2]
    Y[1] = x_past[3]
    Y[2] = g(x_past[0], x_past[1])
    return Y

def h_obs_matrix(x_past):
    p,q=np.shape(x_past)
    Y=np.zeros((3,q))
    Y[0] = x_past[2]
    Y[1] = x_past[3]
    Y[2] = g(x_past[0], x_past[1])
    return Y

def f(x_past, t=None):
    # physical parameters
    L = 2
    delta = 1
    mu = 1
    # compute dynamic model
    x_future = x_past.copy()
    x_future[0] = x_past[2]
    x_future[1] = x_past[3]
    x_future[2] = - delta * L ** 2 * 0.5*np.sin(0.5*x_past[0]) * np.cos(0.5*x_past[1]) - mu * (x_past[2] + L * 0.5*np.sin(0.5*x_past[1]))
    x_future[3] = - delta * L ** 2 * 0.5*np.sin(0.5*x_past[1]) * np.cos(0.5*x_past[0]) - mu * (x_past[3] - L * 0.5*np.sin(0.5*x_past[0]))
    return x_future


def m(x_past, dT_m):
    # Runge-Kutta (4,5) integration method
    X1 = np.copy(x_past)
    k1 = f(X1)

    X2 = np.copy(x_past + k1 / 2 * dT_m)
    k2 = f(X2)

    X3 = np.copy(x_past + k2 / 2 * dT_m)
    k3 = f(X3)

    X4 = np.copy(x_past + k3 * dT_m)
    k4 = f(X4)

    # return the state in the near future
    x_future = x_past + dT_m / 6. * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_future


def EnKf(n, nb, p, Ne, y_o, R, Q,dT,init_state=np.array([0]),init_incert=np.array([0]),init_used=False,v_true_used=False,v_true=None,pos_true=None,visu_direct=False):
    x_f_enkf = zeros((n, nb))  # forecast state
    P_f_enkf = zeros((n, n, nb))  # forecast error covariance matrix
    x_a_enkf = zeros((n, nb))  # analysed state
    P_a_enkf = zeros((n, n, nb))  # analysed error covariance matrix

    ### Ensemble Kalman filter
    x_a_enkf_tmp = zeros((n, Ne))
    x_f_enkf_tmp = zeros((n, Ne))
    y_f_enkf_tmp = zeros((p, Ne))

    ### Init step
    x_a_enkf[:, 0] = init_state  # initial state
    P_a_enkf[:, :, 0] = init_incert  # initial state covariance
    if init_used:
        for i in range(Ne):
            initial=random.multivariate_normal(init_state, init_incert )
            x_a_enkf_tmp[:, i] = initial
            x_f_enkf_tmp[:, i] = initial
    fst_pause = 1
    for k in range(1, nb):  # forward in time
        print("Step: ",k)
        # prediction step
        for i in range(Ne):
            if v_true_used:
                temp_xf=m(x_a_enkf_tmp[:, i], dT)
                temp_xf[2:]=v_true[:,k]
                x_f_enkf_tmp[:, i] = random.multivariate_normal(temp_xf, Q) # to test with true value of speed
            else :
                x_f_enkf_tmp[:, i] = random.multivariate_normal(m(x_a_enkf_tmp[:, i], dT), Q)

            y_f_enkf_tmp[:, i] = random.multivariate_normal(h_obs(x_f_enkf_tmp[:, i]), R)

        P_f_enkf_tmp = cov(x_f_enkf_tmp)
        # Kalman gain
        # H @ P_f_enkf_tmp @ H.T = cov(h_obs(x_f_enkf_tmp))
        # P_f_enkf_tmp @ H.T = voir screen

        P_f_H_t = (x_f_enkf_tmp - np.mean(x_f_enkf_tmp, axis=1).reshape((-1,1))) @ (h_obs_matrix(x_f_enkf_tmp) - np.mean(h_obs_matrix(x_f_enkf_tmp), axis=1).reshape((-1,1))).T / (Ne - 1)
        H_P_f_H_T = (h_obs_matrix(x_f_enkf_tmp) - np.mean(h_obs_matrix(x_f_enkf_tmp), axis=1).reshape((-1,1))) @ (h_obs_matrix(x_f_enkf_tmp) - np.mean(h_obs_matrix(x_f_enkf_tmp), axis=1).reshape((-1,1))).T / (Ne - 1)
        K = P_f_H_t @ np.linalg.inv(H_P_f_H_T + R)

        # update step
        for i in range(Ne):
            x_a_enkf_tmp[:, i] = x_f_enkf_tmp[:, i] + K @ (y_o[:, k] - y_f_enkf_tmp[:, i])

        P_a_enkf_tmp = cov(x_a_enkf_tmp)
        # store results
        x_f_enkf[:, k] = mean(x_f_enkf_tmp, 1)
        P_f_enkf[:, :, k] = P_f_enkf_tmp
        x_a_enkf[:, k] = mean(x_a_enkf_tmp, 1)
        P_a_enkf[:, :, k] = P_a_enkf_tmp

        if visu_direct :
            x_grd, y_grd = np.meshgrid(np.linspace(-12, 12, 100), np.linspace(-12, 12, 100))
            plt.contourf(x_grd, y_grd, g(x_grd, y_grd))
            cb=plt.colorbar()
            cb.set_label("Bathymetry [m]")
            members_points=plt.scatter(x_a_enkf_tmp[0,:],x_a_enkf_tmp[1,:],color='black',s=4)
            true_points=plt.scatter(pos_true[0,k],pos_true[1,k],color='red')
            courant= psi(x_grd,y_grd,2)
            der_x, der_y = np.gradient(courant,axis=0), np.gradient(courant,axis=1)
            plt.quiver(x_grd[::5,::5], y_grd[::5,::5],der_x[::5,::5], der_y[::5,::5])
            plt.legend(( members_points,true_points),
                       ('Members','True Position'),
                       scatterpoints=1,
                       loc='lower left')
            plt.title("Simulation with EnKF and lower density")
            if fst_pause==1:
                plt.pause(3)
                fst_pause-=1
            plt.pause(0.5)
            plt.clf()
    return x_a_enkf, P_a_enkf

def particle_filter(n,nb, Ne, Q, y_obs, dT,init_state=None,init_used=False,pos_true=None,visu_direct=False):
    #init

    particles = np.zeros((n,Ne,nb))
    resultat = np.zeros((n, nb))
    P_resultat = np.zeros((n, n, nb))
    poids = np.ones((Ne,nb))/Ne
    if init_used:
        for p in range(Ne):
            particles[:,p,0] = random.multivariate_normal(init_state, Q)
            resultat[:, 0] = init_state
    elif not init_used:
        for p in range(Ne):
            particles[:, p, 0] = random.multivariate_normal(np.zeros((n)), Q)
    fst_pause = 1
    #forward in time
    for t in range(1,nb):
        print("Step particle : ",str(t))
        particles_f = np.zeros((n,Ne))
        for p in range(Ne):
            particles_f[:,p] = random.multivariate_normal(m(particles[:, p, t - 1], dT), Q)
            poids[p, t] = exp(-sqrt((y_obs[0,t] - particles[2,p,t])**2 + (y_obs[1,t]- particles[3,p,t])**2 + (y_obs[2,t]- g(particles[0,p,t], particles[1,p,t]))**2))
        poids[:,t] = poids[:,t]/sum(poids[:,t])
        #analyse

        resultat[:,t] = sum(particles_f[:,:] * poids[:,t], axis=1)
        P_resultat[:,:,t] = cov(resultat[:,t])

        #resample
        idx = np.random.choice(Ne, size=Ne, replace=True, p=poids[:,t])
        particles[:,:,t] = particles_f[:,idx]
        poids[:,t] = poids[idx,t]

        if visu_direct :
            x_grd, y_grd = np.meshgrid(np.linspace(-12, 12, 100), np.linspace(-12, 12, 100))
            plt.contourf(x_grd, y_grd, g(x_grd, y_grd))
            cb=plt.colorbar()
            cb.set_label("Bathymetry [m]")
            members_points=plt.scatter(particles_f[0,:],particles_f[1,:],color='black',s=4)
            true_points=plt.scatter(pos_true[0,t],pos_true[1,t],color='red')
            courant= psi(x_grd,y_grd,2)
            der_x, der_y = np.gradient(courant,axis=0), np.gradient(courant,axis=1)
            plt.quiver(x_grd[::5,::5], y_grd[::5,::5],der_x[::5,::5], der_y[::5,::5])
            plt.legend(( members_points,true_points),
                       ('Members','True Position'),
                       scatterpoints=1,
                       loc='lower left')
            plt.title("Simulation with particle filter")
            if fst_pause==1:
                plt.pause(8)
                fst_pause-=1
            plt.pause(0.5)
            plt.clf()

    return particles, resultat, P_resultat
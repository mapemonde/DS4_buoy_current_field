from enkf_particle_filter import *

if __name__ == '__main__':
    ### There we configure the simulation and launch items
    ## modified on 17:07 02/04/2023
    method_used = "Kalman"  # "Particle" ou "Kalman"  
    n = 4
    p = 3
    var_R = 0.01  # m/s
    var_Q = 0.01  # 0.1 for EnKF or   0.01  # for Particle filter
    var_bathy = 0.0000001
    Q = var_Q * np.eye(n, n)  # covariance matrix for model
    R = var_R * np.eye(p, p)  # covariance matrix for observations
    R[2, 2] = var_bathy  # Low variance for bathy
    ### Ensemble Kalman initialization
    # TODO : modify Ne for differents tests
    visu_live = False
    Ne = 200  # number of members or particles depends on the filter
    nb = 300  # number of
    dT = 0.1
    x_initial, y_initial, vx_initial, vy_initial = 0, -6, 3, 0  # m, m, m/s, m/s
    var_x_model, var_y_model, var_vx_model, var_vy_model = 0.05, 0.05, 0.05, 0.05  # m, m, m/s, m/s
    initial_state, initial_incert = np.array([x_initial, y_initial, vx_initial, vy_initial]).T, np.diag(
        [var_x_model, var_y_model, var_vx_model, var_vy_model])

    y_o, X_true = generate_obs(x_initial, y_initial, vx_initial, vy_initial, p, n, nb, dT, R)

    ## Cas en exploitant la vitesse réelle
    # x_a_enkf, P_a_enkf = EnKf(n,nb, p ,Ne, y_o, R, Q,dT,initial_state,initial_incert,init_used=True,v_true_used=True,v_true=X_true[2:,:])

    # Cas réel
    if method_used == "Kalman":
        X_a, P_a = EnKf(n, nb, p, Ne, y_o, R, Q, dT, initial_state, initial_incert, init_used=True, v_true_used=False,
                        pos_true=X_true,visu_direct=visu_live)
    elif method_used == "Particle":
        particles, X_a, P_a = particle_filter(n, nb, Ne, Q, y_o, dT, initial_state, init_used=True,pos_true=X_true,visu_direct=visu_live)
    elif method_used == "Both":
        var_Q_enkf = 0.1  # for EnKF
        var_Q_particle = 0.01  # for Particle filter
        Q_enkf = var_Q_enkf * np.eye(n, n)  # covariance matrix for model
        Q_particle = var_Q_particle * np.eye(n, n)  # covariance matrix for model
        X_a_enkf, P_a_enkf = EnKf(n, nb, p, Ne, y_o, R, Q_enkf, dT, initial_state, initial_incert, init_used=True,
                                  v_true_used=False, pos_true=X_true)
        particles, X_a_particle, P_a_particle = particle_filter(n, nb, Ne, Q_particle, y_o, dT, initial_state,
                                                                init_used=True)
    # plot_density(particles, X_true)
    # plot_speeds(X_true, y_o, X_a)
    plot_traj(X_true, X_a)
    # plot_composantes(X_true, y_o, X_a, P_a,nb)
    # plot_bathy_time(y_o, X_a, P_a, nb)

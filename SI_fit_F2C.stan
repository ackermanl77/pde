
functions {
 real[] SIP(real t,       // time
               real[] y,     // system state {prey, predator}
               real[] params, // parameters
               real[] x_r,   // unused data
               int[] x_i) {
                 
    real dy_dt[3];
    
    real beta_r = params[1];
    real beta_s = params[2];
    real delta  = params[3];
    real Karga  = params[4];
    
    
    real alpha = 0.5;
    real gamma = 0.3;
    real mu_s  = 0.013888889;
    real mu_r  = 0.013888889;
    real mu_p  = 0.012;
    real sigma_p  = 1.2;
//    real K     = 10.146;
    
    dy_dt[1] = beta_r*y[1]*(1- (y[1] + y[2] )/Karga) - alpha*y[1] - delta*y[3]*y[1] - (gamma + mu_s)*y[1];
    dy_dt[2] = beta_s*y[2]*(1- (y[1] + y[2] )/Karga)  +  delta*y[3]*y[1] - (gamma + mu_r)*y[2];
    dy_dt[3] = sigma_p*y[2] - mu_p*y[3];
    
  
    
    return dy_dt;
  }
}

data {
  int<lower = 1> n_obs; // Number of days sampled
  int<lower = 1> n_params; // Number of model parameters
  int<lower = 1> n_difeq; // Number of differential equations in the system
  int<lower = 1> n_sample; // Number of hosts sampled at each time point.
  int<lower = 1> n_fake; // This is to generate "predicted"/"unsampled" data
  

  real y[n_obs, 2]; // The binomially distributed data
  real t0; // Initial time point (zero)
  real ts[n_obs]; // Time points that were sampled
  
  real fake_ts[n_fake]; // Time points for "predicted"/"unsampled" data
  
  
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {

  //Support of parameters
  real<lower =  0, upper = 5> beta_r ;
  real<lower =  0, upper = 4> beta_s ;
  real<lower =  0, upper =0.1> delta; 
  real<lower =  1, upper =1e2> Karga;
  real<lower =  0, upper = 1.> s0 ;
  real<lower =  0, upper = 1.> i0; 
  
  //Standard deviation of likelihood distribution
  real<lower = 0> sigma[2];   // measurement errors
  
}

transformed parameters {

  real params[4];   // { alpha, beta, gamma, delta }
  real y_hat[n_obs, n_difeq]; // Output from the ODE solver
//  real y_hat_init[n_difeq];     // initial conditions for both fractions of S and I
  real y_init[n_difeq];     // initial conditions for both fractions of S and I

  y_init[1] = s0;
  y_init[2] = i0;
  y_init[3] = 1.;
  
  params[1] = beta_r;
  params[2] = beta_s;
  params[3] = delta;
  params[4] = Karga;
  
  y_hat = integrate_ode_rk45(SIP, y_init, t0, ts, params, x_r, x_i);
}



model {
  
  // Prior distributions

  beta_r ~ gamma(2.5, 1.0); 
  beta_s ~ gamma(2.5, 1.0); 
  delta ~ gamma(3.0, 1.0); 
  Karga ~ uniform(1.0, 1.0e4);
  s0    ~ uniform(0.0, 1.0  );
  i0    ~ uniform(0.0, 1.0  );
  sigma ~ normal(0, 1);
  
  for (k in 1:2) {

     y[ , k] ~ lognormal(log(y_hat[, k]), sigma[k]);
  }


}

generated quantities {


  // Generate predicted data over the whole time series:
  real fake[n_fake, n_difeq];
 
  fake = integrate_ode_rk45(SIP, y_init, t0, fake_ts, params, x_r, x_i);

  
}

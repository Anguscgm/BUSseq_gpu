#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h>
#include <unistd.h>

__constant__ int d_n_b[128];
__constant__ int d_mu_nu[128];

//Define some hyperparameters for convenience and clarity.
#define p_init_bound 0.5
#define tau0_init 0.005
#define gamma0_init 0
#define gamma1_init -0.1
#define phi_init 5

#define xi_pi 2
#define sigma2_alpha 5
#define sigma2_nu 5
#define sigma2_delta 5
#define sigma2_gamma0 3
#define kappa_gamma1 0.001
#define tau_gamma1 0.01
#define kappa_phi 1
#define tau_phi 0.1
#define a_p 1
#define b_p 3
#define a_tau0 2
#define b_tau0 0.01
#define tau1 100.0

__device__ double rgamma(curandState_t* state, double a, double scale){
    /* Constants : */
    const double sqrt32 = 5.656854;
    const double exp_m1 = 0.36787944117144232159;/* exp(-1) = 1/e */

    /* Coefficients q[k] - for q0 = sum(q[k]*a^(-k))
     * Coefficients a[k] - for q = q0+(t*t/2)*sum(a[k]*v^k)
     * Coefficients e[k] - for exp(q)-1 = sum(e[k]*q^k)
     */
    const double q1 = 0.04166669;
    const double q2 = 0.02083148;
    const double q3 = 0.00801191;
    const double q4 = 0.00144121;
    const double q5 = -7.388e-5;
    const double q6 = 2.4511e-4;
    const double q7 = 2.424e-4;

    const double a1 = 0.3333333;
    const double a2 = -0.250003;
    const double a3 = 0.2000062;
    const double a4 = -0.1662921;
    const double a5 = 0.1423657;
    const double a6 = -0.1367177;
    const double a7 = 0.1233795;

    /* State variables [FIXME for threading!] :*/
    double aa = 0.;
    double aaa = 0.;
    double s, s2, d;    /* no. 1 (step 1) */
    double q0, b, si, c;/* no. 2 (step 4) */

    double e, p, q, r, t, u, v, w, x, ret_val;

    if(scale == 0||a == 0) return 0;

    curandState_t thread_state = *state;
    
    if (a < 1.) { /* GS algorithm for parameters a < 1 */
        e = 1.0 + exp_m1 * a;
        while (1) {
            p = e * curand_uniform_double(&thread_state);
            if (p >= 1.0) {
            x = -log((e - p) / a);
            if (-log(curand_uniform_double(&thread_state)) >= (1.0 - a) * log(x))
                break;
            } else {
                x = exp(log(p) / a);
                if (-log(curand_uniform_double(&thread_state)) >= x)
                    break;
            }
        }
        *state = thread_state;
        return scale * x;
    }

    /* --- a >= 1 : GD algorithm --- */

    /* Step 1: Recalculations of s2, s, d if a has changed */
    if (a != aa) {
        aa = a;
        s2 = a - 0.5;
        s = sqrt(s2);
        d = sqrt32 - s * 12.0;
    }
    /* Step 2: t = standard normal deviate,
               x = (s,1/2) -normal deviate. */

    /* immediate acceptance (i) */
    t = curand_normal_double(&thread_state);
    x = s + 0.5 * t;
    ret_val = x * x;
    if (t >= 0.0){
        *state = thread_state;
        return scale * ret_val;
    }

    /* Step 3: u = 0,1 - uniform sample. squeeze acceptance (s) */
    u = curand_uniform_double(&thread_state);
    if (d * u <= t * t * t){
        *state = thread_state;
        return scale * ret_val;
    }

    /* Step 4: recalculations of q0, b, si, c if necessary */

    if (a != aaa) {
        aaa = a;
        r = 1.0 / a;
        q0 = ((((((q7 * r + q6) * r + q5) * r + q4) * r + q3) * r
               + q2) * r + q1) * r;

        /* Approximation depending on size of parameter a */
        /* The constants in the expressions for b, si and c */
        /* were established by numerical experiments */

        if (a <= 3.686) {
            b = 0.463 + s + 0.178 * s2;
            si = 1.235;
            c = 0.195 / s - 0.079 + 0.16 * s;
        } else if (a <= 13.022) {
            b = 1.654 + 0.0076 * s2;
            si = 1.68 / s + 0.275;
            c = 0.062 / s + 0.024;
        } else {
            b = 1.77;
            si = 0.75;
            c = 0.1515 / s;
        }
    }
    /* Step 5: no quotient test if x not positive */

    if (x > 0.0) {
        /* Step 6: calculation of v and quotient q */
        v = t / (s + s);
        if (fabs(v) <= 0.25)
            q = q0 + 0.5 * t * t * ((((((a7 * v + a6) * v + a5) * v + a4) * v
                          + a3) * v + a2) * v + a1) * v;
        else
            q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);


        /* Step 7: quotient acceptance (q) */
        if (log(1.0 - u) <= q){
            *state = thread_state;
            return scale * ret_val;
        }
    }

    while (1) {
        /* Step 8: e = standard exponential deviate
         *    u =  0,1 -uniform deviate
         *    t = (b,si)-double exponential (laplace) sample */
        e = -log(curand_uniform_double(&thread_state));
        u = curand_uniform_double(&thread_state);
        u = u + u - 1.0;
        if (u < 0.0)
            t = b - si * e;
        else
            t = b + si * e;
        /* Step     9:  rejection if t < tau(1) = -0.71874483771719 */
        if (t >= -0.71874483771719) {
            /* Step 10:     calculation of v and quotient q */
            v = t / (s + s);
            if (fabs(v) <= 0.25)
                q = q0 + 0.5 * t * t *
                    ((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v
                      + a2) * v + a1) * v;
            else
                q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);
            /* Step 11:     hat acceptance (h) */
            /* (if q not positive go to step 8) */
            if (q > 0.0) {
                w = expm1(q);
            /*  ^^^^^ original code had approximation with rel.err < 2e-7 */
            /* if t is rejected sample again at step 8 */
            if (c * fabs(u) <= w * exp(e - 0.5 * t * t))
                break;
            }
        }
    } /* repeat .. until  `t' is accepted */
    x = s + 0.5 * t;
    *state = thread_state;
    return scale * x * x;
}

__device__ int rnbinom(curandState_t* state, double mu, double phi){
    curandState_t thread_state = *state;
    double scale = mu / phi;
    // Sample Gamma dist.
    double gamma = rgamma(&thread_state, phi, scale);
    
    // Sample Poisson dist.
    int pois = curand_poisson(&thread_state, gamma);
    
    *state = thread_state;
    return (pois);
}

__device__ double rbeta(curandState_t* state, double alpha, double beta){
    curandState_t thread_state = *state;
    double A = rgamma(&thread_state, alpha, 1);
    double B = rgamma(&thread_state, beta, 1);
    *state = thread_state;
    return (A/(A+B));
}

__global__ void initialize_curand(int seed, curandState_t* d_states, int rand_len){
    int pos = blockIdx.x*blockDim.x + threadIdx.x;
    if (pos<rand_len)
        curand_init(seed, pos, 0, &d_states[pos]);
}

template <class dataType>
__device__ void warpReduce(volatile dataType *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template <class dataType>
__global__ void sum_on_gpu(dataType *arr, dataType *sum, int n_elem){
    extern __shared__ double smem[];
    dataType *block_level_mem = (dataType *)smem;
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    block_level_mem[tid] = 0;
    while (pos < n_elem){
        block_level_mem[tid] += arr[pos];
        pos += stride;
    }
    __syncthreads();
    if (tid < 512) 
        block_level_mem[tid] += block_level_mem[tid + 512];
    __syncthreads();
    if (tid < 256) 
        block_level_mem[tid] += block_level_mem[tid + 256];
    __syncthreads();
    if (tid < 128) 
        block_level_mem[tid] += block_level_mem[tid + 128];
    __syncthreads();
    if (tid < 64) 
        block_level_mem[tid] += block_level_mem[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce(block_level_mem, tid);
    if(tid == 0)
        *sum = block_level_mem[0];
}

__global__ void selective_mean_on_gpu(double* arr, double* mean, int* count, int n_elem){
    extern __shared__ double block_level_mem[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    block_level_mem[tid] = 0;
    while(pos<n_elem){
        block_level_mem[tid] += arr[pos];
        pos += stride;
    }
    __syncthreads();
    if (tid < 512) 
        block_level_mem[tid] += block_level_mem[tid + 512];
    __syncthreads();
    if (tid < 256) 
        block_level_mem[tid] += block_level_mem[tid + 256];
    __syncthreads();
    if (tid < 128) 
        block_level_mem[tid] += block_level_mem[tid + 128];
    __syncthreads();
    if (tid < 64) 
        block_level_mem[tid] += block_level_mem[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce<double>(block_level_mem, tid);
    if(tid == 0){
        if (*count == 0)
            *mean = 0;
        else
            *mean = block_level_mem[0]/count[0];
    }
}

__global__ void mean_on_gpu(double *arr, double *mean, int n_elem){
    extern __shared__ double block_level_mem[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    block_level_mem[tid] = 0;
    while(pos<n_elem){
        block_level_mem[tid] += arr[pos];
        pos += stride;
    }
    __syncthreads();
    if (tid < 512) 
        block_level_mem[tid] += block_level_mem[tid + 512];
    __syncthreads();
    if (tid < 256) 
        block_level_mem[tid] += block_level_mem[tid + 256];
    __syncthreads();
    if (tid < 128) 
        block_level_mem[tid] += block_level_mem[tid + 128];
    __syncthreads();
    if (tid < 64) 
        block_level_mem[tid] += block_level_mem[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce<double>(block_level_mem, tid);
    if(tid == 0)
        *mean = block_level_mem[0]/n_elem;
}

__global__ void mean_on_gpu(int *arr, double *mean, int n_elem){
    extern __shared__ double smem[];
    int* block_level_mem = (int*)smem;
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    block_level_mem[tid] = 0;
    while(pos<n_elem){
        block_level_mem[tid] += arr[pos];
        pos += stride;
    }
    __syncthreads();
    if (tid < 512) 
        block_level_mem[tid] += block_level_mem[tid + 512];
    __syncthreads();
    if (tid < 256) 
        block_level_mem[tid] += block_level_mem[tid + 256];
    __syncthreads();
    if (tid < 128) 
        block_level_mem[tid] += block_level_mem[tid + 128];
    __syncthreads();
    if (tid < 64) 
        block_level_mem[tid] += block_level_mem[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce<int>(block_level_mem, tid);
    if(tid == 0)
        *mean = block_level_mem[0]/(double)n_elem;
}

__device__ int get_batch(int i){
    int b = 0;
    int temp = d_n_b[b];
    while (i >= temp){
        b++;
        temp += d_n_b[b];
    }
    return (b);
}

__global__ void fill_raw_means(int* d_Y, double* d_delta, int* d_W, double* d_temp_double, int* d_temp_int, int k, int g, int G, int n_b){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    //Please note that input pointers are adjusted by the sample offset.
    if (i < n_b){
        if (d_W[i] == k){
            d_temp_double[i] = log(1 + d_Y[i*G + g] / exp(d_delta[i]));
            d_temp_int[i] = 1;
        }
        else{
            d_temp_double[i] = 0;
            d_temp_int[i] = 0;
        }
    }
}

__global__ void first_gamma(double* d_gamma, int B){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < B){
        d_gamma[pos] = gamma0_init;
        d_gamma[B + pos] = gamma1_init;
    }
}

__global__ void first_pi(double* d_pi, int K, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound){
        d_pi[pos] = (pos % K +1)/(K*(K+1)/2.0); // pos % K is k = 0,1,2,...,K-1
    }
}

__global__ void first_nu_phi(double* d_raw_means, double* d_nu, double* d_phi, int G, int K, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<bound){
        d_phi[pos] = phi_init;
        int b = pos/G;
        if (b ==0)
            d_nu[pos] = 0;
        else{
            int g = pos % G;
            double temp1 = d_raw_means[b*K*G + g];
            double temp2 = d_raw_means[g];
            for (int k=1; k<K; k++){
                temp1 += d_raw_means[(b*K + k)*G + g];
                temp2 += d_raw_means[k*G + g];
            }
            d_nu[pos] = (temp1 - temp2)/K;
        }
    }
}

__global__ void first_delta_W(curandState_t* d_states, double* d_pi, int* d_sum_per_cell, double* d_delta, int* d_W, int N, int K){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<N){
    //W
        int b = get_batch(pos);
        double u = curand_uniform_double(&d_states[pos]);
        
        int k = 0;
        double temp = d_pi[b*K];
        while (u>temp){
            k++;
            temp += d_pi[b*K + k];
        }
        d_W[pos] = k;
    
    //delta
        int ref_pos = 0;
        while (b>0){
            b--;
            ref_pos += d_n_b[b];
        }
        d_delta[pos] = log((double)d_sum_per_cell[pos]) - log((double)d_sum_per_cell[ref_pos]);
    }
}

__global__ void first_beta_L(curandState_t* d_states, double* d_raw_means, double log_rat_base, 
                                    double* d_beta, int* d_L, int G, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    //beta and L
    if (pos < bound){
        if (pos < G){
            d_beta[pos] = 0;
            d_L[pos] = 0;
        }
        else{
            int g = pos % G;
            d_beta[pos] = d_raw_means[pos] - d_raw_means[g]; //pos is k*G + g
            
            double log_rat = log_rat_base + pow(d_beta[pos],2.0)*(1/tau0_init-1/tau1)/2;
            d_L[pos] = (curand_uniform_double(&d_states[pos]) > (1/(1+exp(log_rat))));
        }
    }
}
            
__global__ void first_Z(int* d_Z, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if(pos<bound)
        d_Z[pos] = 0;
}

__global__ void fill_mu_nu(int* d_Y_special, double* d_delta_special, double* d_temp_double, int G, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<bound){
        int i = pos/G;
        d_temp_double[pos] = log(1+d_Y_special[pos]/exp(d_delta_special[i]));
    }
}

__global__ void first_mu_nu(double* d_mean, int bound){
    int tid = threadIdx.x;
    if (tid < bound)
        d_mean[tid+1] -= d_mean[0];
}

__global__ void first_mu_delta(int* d_temp_int, int* d_count, double* d_mu_delta, int N){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<N){
        int b = get_batch(pos);
        d_mu_delta[pos] = log((double)d_count[pos]) - log((double)d_temp_int[b]);
    }
}

__global__ void update_Z_X(curandState_t* d_states, int* d_Y, int* d_W, double* d_alpha, double* d_beta, double* d_nu,
                        double* d_delta, double* d_gamma, double* d_phi,
                        int* d_Z, int* d_X, int B, int N, int G){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<N*G){ //pos is i*G + g
        curandState_t thread_state = d_states[pos];
        int i = pos/G;
        int b = get_batch(i);
        if(d_Y[pos] == 0){
            if(d_X[pos] == 0)
                d_Z[pos] = (curand_uniform_double(&thread_state)>1/(1+exp(d_gamma[b])));
            else
                d_Z[pos] = 1;
            if(d_Z[pos] == 1){
                int g = pos - i*G;
                int k = d_W[i];
                double log_mu = d_alpha[g] + d_beta[k*G+g] + d_nu[b*G + g] + d_delta[i];
                int new_x = rnbinom(&thread_state, exp(log_mu),d_phi[b*G + g]);
                double u =curand_uniform_double(&thread_state);
                //Potential danger here, though not a lot.
                if(u<=(1+exp(-d_gamma[b]-d_gamma[B+b]*d_X[pos])/(1+exp(-d_gamma[b]-d_gamma[B+b]*new_x))))
                    d_X[pos] = new_x;
            }
            else
                d_X[pos] = 0;
        }
        d_states[pos] = thread_state;
    }
}

__global__ void propose_gamma(curandState_t* d_states, double* d_proposed_gamma, double* d_gamma,int B){
    int b = threadIdx.x + blockIdx.x*blockDim.x;
    if(b<B){
        curandState_t thread_state = d_states[b];
        d_proposed_gamma[b] = curand_normal_double(&thread_state)*0.1 + d_gamma[b];
        d_proposed_gamma[B+b] = -rgamma(&thread_state, -10*d_gamma[B+b],0.1);
        d_states[b] = thread_state;
    }
}

__global__ void fill_prop_gamma0(double* d_proposed_gamma, double* d_gamma, int* d_Z_special, int* d_X_special,
                                double* d_temp_double, int b, int B, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;//bound is n_b*G + g.
    if (pos < bound){
        double proposal, previous;
        double new_gamma0 = d_proposed_gamma[b];
        double prev_gamma0 = d_gamma[b];
        double gamma1 = d_gamma[B+b];
        int X = d_X_special[pos];
        int Z = d_Z_special[pos];
        //Proposal
        double temp = new_gamma0 + gamma1*X;
        if (temp>0)
            proposal = new_gamma0*Z - temp - log(1+exp(-temp));
        else
            proposal = new_gamma0*Z - log(1+exp(temp));
        //Previous
        temp = prev_gamma0 + gamma1*X;
        if (temp>0)
            previous = prev_gamma0*Z - temp - log(1+exp(-temp));
        else
            previous = prev_gamma0*Z - log(1+exp(temp));
        
        d_temp_double[pos] = proposal - previous;
    }
}

__global__ void fill_prop_gamma1(double* d_proposed_gamma, double* d_gamma, int* d_Z_special, int* d_X_special,
                                double* d_temp_double, int b, int B, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;//bound is n_b*G + g.
    if (pos < bound){
        double proposal, previous;
        double new_gamma1 = d_proposed_gamma[B+b];
        double prev_gamma1 = d_gamma[B+b];
        double gamma0 = d_gamma[b];
        int X = d_X_special[pos];
        int Z = d_Z_special[pos];
        //Proposal
        double temp = gamma0 + new_gamma1*X;
        if (temp>0)
            proposal = new_gamma1*X*Z - temp - log(1+exp(-temp));
        else
            proposal = new_gamma1*X*Z - log(1+exp(temp));
        //Previous
        temp = gamma0 + prev_gamma1*X;
        if (temp>0)
            previous = prev_gamma1*X*Z - temp - log(1+exp(-temp));
        else
            previous = prev_gamma1*X*Z - log(1+exp(temp));
        
        d_temp_double[pos] = proposal - previous;
    }
}

__global__ void update_gamma0(curandState_t* d_states, double* d_proposed_gamma, double* d_log_rho, double* d_gamma, int B){
    int b = threadIdx.x + blockIdx.x*blockDim.x;
    if(b<B){
        double logr_gamma0_prior = (pow(d_gamma[b], 2.0) - pow(d_proposed_gamma[b],2.0))/(2*sigma2_gamma0);
        
        if (log(curand_uniform_double(&d_states[b])) <= (logr_gamma0_prior + d_log_rho[b]))
            d_gamma[b] = d_proposed_gamma[b];
    }
}

__global__ void update_gamma1(curandState_t* d_states, double* d_proposed_gamma, double* d_log_rho, double* d_gamma, int B){
    int b = threadIdx.x + blockIdx.x*blockDim.x;
    if(b<B){
        double prev_gamma1 = -d_gamma[B+b];
        double new_gamma1 = -d_proposed_gamma[B+b];
        double logr_gamma1 = (kappa_gamma1 - 1)*(log(new_gamma1) - log(prev_gamma1))
                            + tau_gamma1*(prev_gamma1 - new_gamma1)
                            - lgamma(10*new_gamma1) + (10*new_gamma1 - 1)*log(prev_gamma1) + 10*new_gamma1*(log(10.0)+1)
                            + lgamma(10*prev_gamma1) - (10*prev_gamma1 - 1)*log(new_gamma1) - 10*prev_gamma1*(log(10.0)+1);
        if (log(curand_uniform_double(&d_states[b])) <= (logr_gamma1 + d_log_rho[b]))
            d_gamma[B+b] = -new_gamma1;
    }
}

__global__ void propose_alpha(curandState_t* d_states, double* d_proposed_alpha, double* d_alpha, int G){
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g<G)
        d_proposed_alpha[g] = curand_normal_double(&d_states[g])*0.1 + d_alpha[g];
}

__global__ void fill_prop_alpha(double* d_proposed_alpha, double* d_alpha, double* d_beta, double* d_nu, double* d_delta,
                            double* d_phi, int* d_W, int* d_X, double* d_temp_double, int g, int B, int N, int G){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<N){
        int k = d_W[i];
        int b = get_batch(i);
        double phi = d_phi[b*G + g];
        double mu = d_beta[k*G + g] + d_nu[b*G + g] + d_delta[i];//everything except alpha
        int X = d_X[i*G + g];
        double prev_alpha = d_alpha[g];
        double new_alpha = d_proposed_alpha[g];
        d_temp_double[i] = (log(phi + exp(prev_alpha + mu))
                        - log(phi + exp(new_alpha + mu)))
                        * (phi + X) + (new_alpha - prev_alpha)*X;
    }
}

__global__ void update_alpha(curandState_t* d_states, double* d_proposed_alpha, double* d_log_rho, double* d_mu_alpha, double* d_alpha, int G){
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g<G){
        double logr_alpha = (pow(d_alpha[g] - d_mu_alpha[g], 2.0)- pow(d_proposed_alpha[g] - d_mu_alpha[g], 2.0))/(2*sigma2_alpha);
        if(log(curand_uniform_double(&d_states[g])) <= d_log_rho[g] + logr_alpha)
            d_alpha[g] = d_proposed_alpha[g];
    }
}

__global__ void update_L(curandState_t* d_states, double* d_p, double* d_beta_special, int* d_L_special, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound){
        double p = d_p[0];
        double tau0 = d_p[1];
        //Use [pos+G] to skip k=0.
        double log_odd = log(p) - log(1-p) 
                        - (log(tau1) - log(tau0))/2.0
                        + pow(d_beta_special[pos],2.0)*(1/(2*tau0) - 1/(2*tau1));
        if(curand_uniform_double(&d_states[pos]) > 1/(1+exp(log_odd)))
            d_L_special[pos] = 1;
        else
            d_L_special[pos] = 0;
    }
}

__global__ void update_p(curandState_t* d_states, int* d_count, double* d_p, int G, int K){
    d_p[0] = rbeta(d_states, *d_count + a_p, G*(K-1) - *d_count + b_p);
}

__global__ void fill_I_L_beta_sq(int* d_L_special, double* d_beta_special, double* d_temp_double, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound){//pos is (k-1)*G + g
        if (d_L_special[pos] == 0)
            d_temp_double[pos] = pow(d_beta_special[pos], 2.0);
        else
            d_temp_double = 0;
    }
}

__global__ void update_tau0(curandState_t* d_states, int* d_count, double* d_p, int G, int K){
    double a = a_tau0 + (G*(K-1) - d_count[0])/2.0;
    double b = b_tau0 + d_p[1]/2.0;
    d_p[1] = 1/rgamma(d_states, a, 1/b); //use 0 position
}

__global__ void propose_beta(curandState_t* d_states, double* d_proposed_beta, double* d_beta_special, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound)
        d_proposed_beta[pos] = curand_normal_double(&d_states[pos])*0.1 + d_beta_special[pos];
}

__global__ void fill_prop_beta(double* d_proposed_beta, double* d_beta, double* d_alpha, double* d_nu, double* d_delta,
                            double* d_phi, int* d_W, int* d_X, double* d_temp_double, int g, int k, int N, int G){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N){
        if (d_W[i] == k){
            int b = get_batch(i);
            double mu = d_alpha[g] + d_nu[b*G + g] + d_delta[i]; //mu other than beta
            double new_beta = d_proposed_beta[(k-1)*G + g];
            double prev_beta = d_beta[k*G + g];
            double phi = d_phi[b*G + g];
            int X = d_X[i*G + g];
            d_temp_double[i] = (new_beta - prev_beta)*X
                            + (phi + X) *(log(phi + exp(mu + prev_beta))
                            - log(phi + exp(mu + new_beta)));
        }
        else
            d_temp_double[i] = 0;
    }
}

__global__ void update_beta(curandState_t* d_states, double* d_proposed_beta, double* d_log_rho, int* d_L_special, double* d_beta_special, double* d_p, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound){
        double logr_beta;
        if (d_L_special[pos] == 1) //pos is [(k-1)*G+g]
            logr_beta = (pow(d_beta_special[pos],2.0) - pow(d_proposed_beta[pos], 2.0))/(2*tau1);
        else
            logr_beta = (pow(d_beta_special[pos],2.0) - pow(d_proposed_beta[pos], 2.0))/(2*d_p[1]);
        if (log(curand_uniform_double(&d_states[pos])) <= d_log_rho[pos] + logr_beta)
            d_beta_special[pos] = d_proposed_beta[pos];
    }
}

__global__ void propose_nu(curandState_t* d_states, double* d_proposed_nu, double* d_nu_special, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound)
        d_proposed_nu[pos] = curand_normal_double(&d_states[pos])*0.1 + d_nu_special[pos];
}

__global__ void fill_prop_nu(double* d_proposed_nu, double* d_nu, double* d_alpha, double* d_beta, double* d_delta, double* d_phi, 
                            int* d_W_special, int* d_X_special, double* d_temp_double, int b, int g, int G, int n_b){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n_b){
        double new_nu = d_proposed_nu[(b-1)*G + g];
        double prev_nu = d_nu[b*G + g];
        double mu = d_alpha[g] + d_beta[d_W_special[i]*G + g] + d_delta[i];//everything in mu acept nu
        double phi = d_phi[b*G + g];
        int X = d_X_special[i*G + g];
        d_temp_double[i] = (new_nu - prev_nu)*X
                            + (phi + X)*(log(phi + exp(mu + prev_nu))
                            - log(phi + exp(mu + new_nu)));
    }
}

__global__ void update_nu(curandState_t* d_states, double* d_proposed_nu, double* d_log_rho, double* d_nu_special, int G, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound){ //pos is (b-1)*G + g
        double logr_nu = (pow(d_nu_special[pos] - d_mu_nu[pos/G], 2.0) - pow(d_proposed_nu[pos] - d_mu_nu[pos/G], 2.0))/(2*sigma2_nu);
        if (log(curand_uniform_double(&d_states[pos])) <= logr_nu + d_log_rho[pos])
            d_nu_special[pos] = d_proposed_nu[pos];
    }
}

__global__ void propose_delta(curandState_t* d_states, double* d_proposed_delta, double* d_delta, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<N){
        d_proposed_delta[i] = curand_normal_double(&d_states[i])*0.1 + d_delta[i];
    }
}

__global__ void fill_prop_delta(double* d_proposed_delta, double* d_delta, double* d_alpha, double* d_beta, double* d_nu, double* d_phi,
                                int* d_W, int* d_X, double* d_temp_double, int i, int G){
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g<G){
        int b = get_batch(i);
        double mu = d_alpha[g] + d_beta[d_W[i]*G + g] + d_nu[b*G + g];
        double phi = d_phi[b*G + g];
        double new_delta = d_proposed_delta[i];
        double prev_delta = d_delta[i];
        int X = d_X[i*G + g];
        d_temp_double[i] = (new_delta - prev_delta)*X 
                            + (phi + X)*(log(phi + exp(mu + prev_delta))
                            - log(phi + exp(mu + new_delta)));
    }
    
}

__global__ void update_delta(curandState_t* d_states, double* d_proposed_delta, double* d_log_rho, double* d_mu_delta, double* d_delta, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<N){
        double logr_delta = (pow(d_delta[i] - d_mu_delta[i], 2.0)-pow(d_proposed_delta[i] - d_mu_delta[i], 2.0))/(2*sigma2_delta);
        if (log(curand_uniform_double(&d_states[i])) <= logr_delta + d_log_rho[i])
            d_delta[i] = d_proposed_delta[i];
    }
}

__global__ void propose_phi(curandState_t* d_states, double* d_proposed_phi, double* d_phi, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<bound){ //b*G+g
        d_proposed_phi[pos] = rgamma(&d_states[pos],d_phi[pos],1);
    }
}

__global__ void fill_prop_phi(double* d_proposed_phi_special, double* d_phi_special, double* d_alpha_special, double* d_beta_special,
                            double* d_nu_special, double* d_delta_special, int* d_W_special, int* d_X_special, double* d_temp_double,  int G){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<d_n_b[b]){
        double new_phi = *d_proposed_phi_special;
        double prev_phi = *d_phi_special;
        int X = d_X_special[i*G];//offset by sample_index*G + g
        //offset alpha by g, beta by g, W by sample_index, nu by bg, delta by sample_index
        double eta = exp(*d_alpha_special + d_beta_special[d_W_special[i]*G] + d_nu_special + d_delta_special[i]);
        d_temp_double[i] = lgamma(new_phi + X) + new_phi*log(new_phi)
                            - lgamma(new_phi) - (new_phi + X)*log(new_phi + eta)
                            + lgamma(prev_phi) + (prev_phi + X)*log(prev_phi + eta)
                            - lgamma(prev_phi + X) - prev_phi*log(prev_phi);
    }
}

__global__ void update_phi(curandState_t* d_states, double* d_proposed_phi, double* d_log_rho, double* d_phi, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos< bound){ //b*G+g
        double new_phi = d_proposed_phi[pos];
        double prev_phi = d_phi[pos];
        double logr_phi = (kappa_phi - prev_phi)*log(new_phi)
                            - (kappa_phi - new_phi)*log(prev_phi)
                            + (1 - tau_phi)*(new_phi - prev_phi)
                            + lgamma(prev_phi) - lgamma(new_phi);
        if (log(curand_uniform_double(&d_states[pos])) <= logr_phi + d_log_rho[pos])
            d_phi[pos] = new_phi;
    }
}

__global__ void propose_W(curandState_t* d_states, int* d_proposed_W, int N, int K){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<N){
        double temp = curand_uniform_double(&d_states[i])*K;
        int k = 0;
        while (temp > k+1)
            k++;
        d_proposed_W[i] = k;
    }
}

__global__ void fill_prop_W(int* d_proposed_W, int* d_W, double* d_alpha, double* d_beta, double* d_nu, double* d_delta,
                            double* d_phi, int* d_X, double* d_temp_double, int i, int G){
    int g = threadIdx.x + blockIdx.x*blockDim.x;
    if (g<G){
        int bg = get_batch(i)*G + g;
        double new_beta = d_beta[d_proposed_W[i]*G+g];
        double prev_beta = d_beta[d_W[i]*G+g];
        double mu = d_alpha[g] + d_nu[bg] + d_delta[i];//everything except beta
        double phi = d_phi[bg];
        d_temp_double[g] = (new_beta-prev_beta)*d_X[i*G+g]
                            + phi*(log(phi + exp(mu + prev_beta))
                            - log(phi + exp(mu + new_beta)));
    }
}

__global__ void update_W(curandState_t* d_states, int* d_proposed_W, double* d_log_rho, double* d_pi, int* d_W, int N, int K){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<N){
        int b = get_batch(i);
        int new_W = d_proposed_W[i];
        if (log(curand_uniform_double(&d_states[i])) <= log(d_pi[b*K + new_W]/d_pi[b*K + d_W[i]]) + d_log_rho[i])
            d_W[i] = new_W;
    }
}

__global__ void fill_I_W(int* d_W_special, int* d_temp_int, int k, int n_b){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<n_b){
        if(d_W_special[pos] == k)
            d_temp_int[pos] = 1;
        else
            d_temp_int[pos] = 0;
    }
}

__global__ void update_pi(curandState_t* d_states, int* d_count, double* d_pi, int K){
    //This is effectively just a dirichlet sampler
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    if (tid < K){
        curandState_t thread_state = d_states[tid];
        double d =  d_count[tid] + xi_pi - 1/3.0;
        while(1){
            double z = curand_normal_double(&thread_state);
            double v = pow((1 + z/(3*sqrt(d))),3.0);
            if(v > 0 && log(curand_uniform_double(&thread_state))< 0.5*z*z + d *(1-v + log(v))){
                d_states[tid] = thread_state; //Send back the latest state.
                d_pi[tid] = d*v;
                break;
            }
        }
        if(tid==0){
            smem[0] = 0;
            for(int i=0; i<K; i++)
                smem[0] += d_pi[i];
        }
        __syncthreads();
        d_pi[tid] /= smem[0];
    }
}

__global__ void store_gamma(double* d_gamma, double* d_preserved_gamma, int iter, int n_preserved, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound) // bound is 0/1*B + b
        d_preserved_gamma[pos*n_preserved + iter] = d_gamma[pos];
}

__global__ void store_alpha(double* d_alpha, double* d_preserved_alpha, int iter, int n_preserved, int G){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<G) //pos is g
        d_preserved_alpha[pos*n_preserved + iter] = d_alpha[pos];
}

__global__ void store_L_beta(int* d_L_special, int* d_preserved_L, double* d_beta_special, double* d_preserved_beta, int iter, int n_preserved, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound){// pos is (k-1)*G + g
        //Note that d_L and d_beta has been offset.
        int new_pos = pos*n_preserved + iter;
        d_preserved_L[new_pos] = d_L_special[pos];
        d_preserved_beta[new_pos] = d_beta_special[pos];
    }
}

__global__ void store_nu_phi(double* d_nu, double* d_preserved_nu, double* d_phi, double* d_preserved_phi, int iter, int n_preserved, int G, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < bound){ //pos is b*G + g
        if(pos>=G)
            d_preserved_nu[(pos-G)*n_preserved + iter] = d_nu[pos];
        d_preserved_phi[pos*n_preserved + iter] = d_phi[pos];
    }
}

__global__ void store_delta_W(double* d_delta, double* d_preserved_delta, int* d_W, int* d_preserved_W, int iter, int n_preserved, int N){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<N){ //pos is i
        int new_pos = pos*n_preserved + iter;
        d_preserved_delta[new_pos] = d_delta[pos];
        d_preserved_W[new_pos] = d_W[pos];
    }
}

__global__ void store_pi(double* d_pi, double* d_preserved_pi, int iter, int n_preserved, int bound){
    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos<bound) //pos is b*K + k
        d_preserved_pi[pos*n_preserved + iter] = d_pi[pos];
}

__global__ void fill_post_W(int* d_preserved_W, int* d_temp_int, int k, int n_preserved){
    int iter = threadIdx.x + blockIdx.x*blockDim.x;
    if (iter<n_preserved)
        d_temp_int[iter] = (d_preserved_W[iter] == k);
}

__global__ void mode_on_gpu(int* d_sum, int* d_post, int length, int types){
    int pos = blockIdx.x*blockDim.x + threadIdx.x;
    if (pos<length){
        int cur_type = 0;
        int cur_max = d_sum[pos];//d_sum in the format of [k*N + i]
        int value;
        for (int k=1; k<types; k++){
            value = d_sum[k*length+pos];
            if(value>cur_max){
                cur_type = k;
                cur_max = value;
            }
        }
        d_post[pos] = cur_type;
    }
}

__global__ void binary_mode_on_gpu(int* d_sum, int* d_post, int length, int max){
    int pos = blockIdx.x*blockDim.x + threadIdx.x;
    if (pos<length)
        d_post[pos] = (d_sum[pos]*2>max);
}

int main(int argc, char **argv){
    // ./BUSseq_gpu -B 4 -N demo_dim.txt -G 3000 -K 5 -s 123 -c demo_count.txt -i 4000 -b 2000 -u 500 -p -o demo_output
    int B = 4; //Number of batches
    int n_b[200] = {300, 300, 200, 200}; // Sample size
    int G = 3000; //Number of genomic locations
    int K = 5; //Number of celltypes
    int seed;
    char count_data[200] = "count_data/demo_count.txt";
    int n_iter = 4000; //Number of iterations
    int n_burnin = 2000; //Number of burn-in iterations
    int n_unchanged = 500;
    int print_preserved = 0; //Default: does not print all preserved iterations.
    char output_file[200] = "demon_output";
    
    char n_file[200];
    int n_file_flag = 0;
    int burnin_flag = 1;
    int unchanged_flag = 1;
    int seed_flag = 1;
    
    int opt;
    
    while ((opt = getopt (argc, argv, "B:N:G:K:s:c:i:b:u:po:")) != -1){
        switch(opt){
            case 'B':
                B = atoi(optarg);
                break;
            case 'N':
                n_file_flag = 1;
                strcpy(n_file, optarg);
                break;
            case 'G':
                G = atoi(optarg);
                break;
            case 'K':
                K = atoi(optarg);
                break;
            case 's':
                seed_flag = 0;
                seed = atoi(optarg);
                break;
            case 'c':
                strcpy(count_data, optarg);
                break;
            case 'i':
                n_iter = atoi(optarg);
                break;
            case 'b':
                burnin_flag = 0;
                n_burnin = atoi(optarg);
                break;
            case 'u':
                unchanged_flag = 0;
                n_unchanged = atoi(optarg);
                break;
            case 'p':
                print_preserved = 1;
                break;
            case 'o':
                strcpy(output_file, optarg);
                break;
            default:
            printf("Error Usage: %s [-b batch] [-n sample_size_file] [-g gene] [-k celltype] [-c count_data_file)] [-i iterations] [-b burnin] [-u unchanged_iterations] [-p print_preserved] [-o output_prefix] \n", argv[0]);
            exit(1);
        }
    }
    if (argc > 1){
        if (n_file_flag){
            FILE* batch_file;
            batch_file = fopen(n_file, "r");
            for (int b=0; b<B; b++)
                fscanf(batch_file, "%d", &n_b[b]);
            fclose(batch_file);
        }
            
        if (burnin_flag){
            n_burnin = n_iter/2;
        }
        
        if (unchanged_flag){
            if (0.3*n_iter > 500)
                n_unchanged = 0.3*n_iter;
            else
                n_unchanged = 500;
        }
    }
    // Get seed
    if (seed_flag){
        struct timeval currentTime;
        gettimeofday(&currentTime, NULL);
        seed = (int) currentTime.tv_usec;
    }
    printf("Seed is %d\n", seed);
    
    //Run checking on user input.
    if(n_burnin > n_iter){
        printf("Burn-in iterations must be less than total number of iterations.\n");
        return (-1);
    }
    if(B<2){
        printf("The batch number must be greater than one.\n");
        return (-1);
    }
    for (int b=0; b<B; b++){
        if (n_b[b]<K){
            printf("The sample size in any batch must be greater than the assumed cell type number.\n");
            return (-1);
        }
    }
    
    //Define some parameters.
    int N = 0;
    for (int b=0; b<B; b++) N += n_b[b];
    cudaMemcpyToSymbol(d_n_b, n_b, B*sizeof(int));
    int n_preserved = n_iter - n_burnin;
    int threads_per_block = 1024;
    
    //Read file.
    printf("Start reading file.\n");
    FILE* myFile;
    myFile = fopen(count_data,"r");
    char gene_names[G][32];
    //Array designed in the way such that the i-th sample's g-th gene is Y[i*G + g]
    int* h_Y = (int *)malloc(N * G * sizeof(int));
    int* d_Y; cudaMalloc(&d_Y, N*G*sizeof(int));
    
    //First, read in the gene names, which is the first row of the file.
    for (int g=0; g<G; g++)
        fscanf(myFile, "%s",  gene_names[g]);
    //Read counts
    for (int i=0; i<N; i++)
        for (int g=0; g<G; g++)
            fscanf(myFile,"%d",&h_Y[i*G + g]);
    fclose(myFile);
    cudaMemcpy(d_Y, h_Y, N*G*sizeof(int), cudaMemcpyHostToDevice);
    free(h_Y);
    
    //1.Initialize
    printf("Start initialization.\n");
    
    //Initialize cuRandStates
    int rand_len = N*G; //rand_len is the length of the longest random variable necessary.
    curandState_t* d_states; cudaMalloc(&d_states, rand_len*sizeof(curandState_t));
    initialize_curand <<< (rand_len-1)/threads_per_block + 1, threads_per_block>>>(seed, d_states, rand_len);
    
    //Initialize first iteration.
    srand(seed);
    double p[2] = {((double)rand()*p_init_bound)/RAND_MAX, tau0_init}; //p and tau0
    double* d_p; cudaMalloc(&d_p, 2*sizeof(double));
    cudaMemcpy(d_p, p, 2*sizeof(double),cudaMemcpyHostToDevice);
    //d_tau is defined as constant memory variables.
    double* d_gamma; cudaMalloc(&d_gamma, B*2*sizeof(double)); //[0/1*B + b]
    double* d_pi; cudaMalloc(&d_pi, K*B*sizeof(double)); //[b*K + k]
    double* d_phi; cudaMalloc(&d_phi, B*G*sizeof(double));//[b*G + g]
    int* d_W; cudaMalloc(&d_W, N*sizeof(int));//[i]
    int* d_sum_per_cell; cudaMalloc(&d_sum_per_cell, N*sizeof(int));
    double* d_delta; cudaMalloc(&d_delta, N*sizeof(double));//[i]
    double* d_alpha; cudaMalloc(&d_alpha, G*sizeof(double));//[g]
    double* d_beta; cudaMalloc(&d_beta, G*K*sizeof(double));//[k*G + g]
    int* d_L; cudaMalloc(&d_L, G*K*sizeof(int));//[k*G + g]
    double log_rat_base = log(p[0]/(1-p[0])) + (log(tau0_init)- log(tau1))/2;
    double* d_nu; cudaMalloc(&d_nu, B*G*sizeof(double));//[b*G + g]
    int* d_Z; cudaMalloc(&d_Z, N*G*sizeof(int));//[i*G + g]
    int* d_X; cudaMalloc(&d_X, N*G*sizeof(int));//[i*G + g]
    
    double* d_mu_alpha; cudaMalloc(&d_mu_alpha, G*sizeof(double));
    double* d_mu_delta; cudaMalloc(&d_mu_delta, N*sizeof(double));
    
    double* d_raw_means; cudaMalloc(&d_raw_means, B*K*G*sizeof(double));//[b*K*G + k*G + g]
    int* d_temp_int;
    if(N>n_preserved)
        cudaMalloc(&d_temp_int, N*sizeof(int));
    else
        cudaMalloc(&d_temp_int, n_preserved*sizeof(int));
    double* d_temp_double; cudaMalloc(&d_temp_double, N*G*sizeof(double));
    int* d_count; cudaMalloc(&d_count, K*N*sizeof(int));
    double* d_mean; cudaMalloc(&d_mean, B*sizeof(double));
    
    for(int i=0; i<N; i++)
        sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (&d_Y[i*G], &d_sum_per_cell[i], G);
    
    first_pi <<<(B*K-1)/ threads_per_block+1, threads_per_block>>> (d_pi, K, B*K);
    first_delta_W <<<(N-1)/threads_per_block+1, threads_per_block>>> (d_states, d_pi, d_sum_per_cell, d_delta, d_W, N, K);
    
    int sample_index = 0;
    for (int b=0; b<B; b++){
        if (b>0)
            sample_index += n_b[b-1];
        for (int k=0; k<K; k++){
            for (int g=0; g<G; g++){
                fill_raw_means <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>> (&d_Y[sample_index*G], &d_delta[sample_index], &d_W[sample_index], d_temp_double, d_temp_int, k, g, G, n_b[b]);
                sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (d_temp_int, d_count, n_b[b]);
                selective_mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_raw_means[b*K*G + k*G + g], d_count, n_b[b]);
            }
        }
    }
    
    first_gamma <<<(B-1)/threads_per_block+1, threads_per_block>>> (d_gamma, B);
    first_nu_phi <<<(B*G-1)/threads_per_block+1, threads_per_block>>> (d_raw_means, d_nu, d_phi, G, K, B*G);
    first_beta_L <<<(G*K-1)/threads_per_block+1, threads_per_block>>> (d_states, d_raw_means, log_rat_base, d_beta, d_L, G, G*K);
    first_Z <<<(N*G-1)/threads_per_block+1, threads_per_block>>> (d_Z, N*G);
    cudaMemcpy(d_alpha, d_raw_means, G*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_X, d_Y, N*G*sizeof(int), cudaMemcpyDeviceToDevice);//Because d_X is initialized to be the same as observed
    
    cudaMemcpy(d_mu_alpha, d_raw_means, G*sizeof(double), cudaMemcpyDeviceToDevice);
    
    sample_index = 0;
    for (int b=0; b<B; b++){
        fill_mu_nu <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>> (&d_Y[sample_index*G], &d_delta[sample_index], d_temp_double, G, n_b[b]*G);
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_mean[b], n_b[b]*G);
        sample_index += n_b[b];
    }
    first_mu_nu <<<1, threads_per_block>>> (d_mean, B-1);
    cudaMemcpyToSymbol(d_mu_nu, &d_mean[1], (B-1)*sizeof(double), cudaMemcpyDeviceToDevice);
    
    int offset = 0;
    for (int b=0; b<B; b++){
        sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (&d_Y[offset*G], &d_count[b], G);
        cudaMemcpy(&d_temp_int[offset],&d_count[b], sizeof(int), cudaMemcpyDeviceToDevice);
        offset++;
        for (int i=1; i<n_b[b]; i++){
            sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (&d_Y[offset*G], &d_temp_int[offset], G);
            offset++;
        }
    }
    first_mu_delta <<< (N-1)/threads_per_block + 1, threads_per_block>>> (d_temp_int, d_count, d_mu_delta, N);
    
    //2.MCMC
    printf("Start MCMC.\n");
    //Declare some arrays before going into MCMC.
    int log_rho_len;
    if (B>K)
        log_rho_len = G*B;
    else
        log_rho_len = G*K;
    if (N>G)
        log_rho_len = log_rho_len/G * N;
    double* d_log_rho; cudaMalloc(&d_log_rho, log_rho_len*sizeof(double)); //To be used for all MH steps.
    //I defined it to be G*K(or B). This is not a mistake, rest assured.
    
    double* d_proposed_gamma; cudaMalloc(&d_proposed_gamma, B*2*sizeof(double));
    double* d_proposed_alpha; cudaMalloc(&d_proposed_alpha, G*sizeof(double));
    double* d_proposed_beta; cudaMalloc(&d_proposed_beta, G*(K-1)*sizeof(double));
    double* d_proposed_nu; cudaMalloc(&d_proposed_nu, G*(B-1)*sizeof(double));
    double* d_proposed_delta; cudaMalloc(&d_proposed_delta, N*sizeof(double));
    double* d_proposed_phi; cudaMalloc(&d_proposed_phi, B*G*sizeof(double));
    int* d_proposed_W; cudaMalloc(&d_proposed_W, N*sizeof(int));
    
    for (int iter=0; iter<n_burnin; iter++){
        //Z and X
        update_Z_X <<<(N*G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_Y, d_W, d_alpha, d_beta, d_nu,
                        d_delta, d_gamma, d_phi,
                        d_Z, d_X, B, N, G);//If there are not enough threads, may have to separate Z and X.
        
        // gamma
        propose_gamma <<<(B-1)/threads_per_block + 1, threads_per_block>>>(d_states, d_proposed_gamma, d_gamma,B);
        
        //Fill gamma0
        sample_index = 0;
        for (int b=0; b<B; b++){
            if (b>0)
                sample_index += n_b[b-1];
            fill_prop_gamma0 <<<(n_b[b]*G-1)/threads_per_block+1, threads_per_block>>> (d_proposed_gamma, d_gamma, &d_Z[sample_index*G],
                                                                                        &d_X[sample_index*G], d_temp_double, b, B, n_b[b]*G);
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[b], n_b[b]*G);
        }
        update_gamma0 <<< (B-1)/threads_per_block + 1, threads_per_block >>> (d_states, d_proposed_gamma, d_log_rho, d_gamma, B);
        
        //Fill gamma1
        sample_index = 0;
        for (int b=0; b<B; b++){
            if (b>0)
                sample_index += n_b[b-1];
            fill_prop_gamma1 <<<(n_b[b]*G-1)/threads_per_block+1, threads_per_block>>> (d_proposed_gamma, d_gamma, &d_Z[sample_index*G],
                                                                                        &d_X[sample_index*G], d_temp_double, b, B, n_b[b]*G);
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[b], n_b[b]*G);
        }
        update_gamma1 <<< (B-1)/threads_per_block + 1, threads_per_block >>> (d_states, d_proposed_gamma, d_log_rho, d_gamma, B);
        
        //alpha
        propose_alpha <<<(G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_alpha, d_alpha, G);
        
        for (int g=0; g<G; g++){
            fill_prop_alpha <<< (N-1)/threads_per_block + 1, threads_per_block >>>(d_proposed_alpha, d_alpha, d_beta, d_nu, d_delta, d_phi,
                                                                                d_W, d_X, d_temp_double, g, B, N, G);
            sum_on_gpu <double> <<< 1, threads_per_block, threads_per_block*sizeof(double) >>>(d_temp_double, &d_log_rho[g], N);
        }
        update_alpha <<< (G-1)/threads_per_block + 1, threads_per_block >>>(d_states, d_proposed_alpha, d_log_rho, d_mu_alpha, d_alpha, G);
        
        //L
        update_L <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_p, &d_beta[G], &d_L[G], G*(K-1));
        
        //p
        if(iter >= n_unchanged){
            //Fill d_count[0] with sum of d_L and d_p[1] with sum of I(d_L==0)*d_beta^2
            //I used d_p[1] to carry the second sum. Its value will be replaced anyway.
            sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (&d_L[G], &d_count[0], G*(K-1));
            fill_I_L_beta_sq <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (&d_L[G], &d_beta[G], d_temp_double, G*(K-1));
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_p[1], G*(K-1));
            update_p <<<1,1>>> (d_states, d_count, d_p, G, K);
            update_tau0 <<<1,1>>> (d_states, d_count, d_p, G, K);
        }
        
        //beta
        propose_beta <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_beta, &d_beta[G], G*(K-1));
        for (int k=1; k<K; k++){
            for (int g=0; g<G; g++){
                fill_prop_beta <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_proposed_beta, d_beta, d_alpha, d_nu, d_delta,
                                                                                d_phi, d_W, d_X, d_temp_double, g, k, N, G);
                sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>>(d_temp_double, &d_log_rho[(k-1)*G + g], N);
            }
        }
        update_beta <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_beta, d_log_rho, &d_L[G], &d_beta[G], d_p, G*(K-1));
        
        //nu
        propose_nu <<<(G*(B-1)-1)/threads_per_block + 1, threads_per_block>>>(d_states, d_proposed_nu, &d_nu[G], G*(B-1));
        sample_index = 0;
        for (int b=1; b<B; b++){
            sample_index += n_b[b-1];
            for (int g=0; g<G; g++){
                fill_prop_nu <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>>(d_proposed_nu, d_nu, d_alpha, d_beta, &d_delta[sample_index], d_phi, 
                                                                                        &d_W[sample_index], &d_X[sample_index*G], d_temp_double, b, g, G, n_b[b]);
                sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[(b-1)*G + g], n_b[b]);
            }
        }
        update_nu <<<(G*(B-1)-1)/threads_per_block+1, threads_per_block>>> (d_states, d_proposed_nu, d_log_rho, &d_nu[G], G, G*(B-1));
        
        //delta
        propose_delta <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_delta, d_delta, N);
        
        for (int i=0; i<N; i++){
            fill_prop_delta <<<(G-1)/threads_per_block + 1, threads_per_block>>> (d_proposed_delta, d_delta, d_alpha, d_beta, d_nu, d_phi,
                                                                                d_W, d_X, d_temp_double, i, G);
            sum_on_gpu <double> <<<(G-1)/threads_per_block + 1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[i], G);
        }
        update_delta <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_delta, d_log_rho, d_mu_delta, d_delta, N);
        
        //phi
        propose_phi <<<(B*G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_phi, d_phi, B*G);
        sample_index = 0;
        int bg = 0;
        for (int b=0; b<B; b++){
            if(b>0)
                sample_index += n_b[b-1];
            for (int g=0; g<G; g++){
                fill_prop_phi <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>> (&d_proposed_phi[bg], &d_phi[bg], &d_alpha[g], &d_beta[g], &d_nu[bg],
                                                                                        &d_delta[sample_index], &d_W[sample_index], &d_X[sample_index*G + g], d_temp_double, G);
                sum_on_gpu <double> <<<1, threads_per_block , threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[bg], n_b[b]);
                bg++;
            }
        }
        update_phi <<<(B*G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_phi, d_log_rho, d_phi, B*G);
        
        //w
        propose_W <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_W, N, K);
        
        for (int i=0; i<N; i++){
            fill_prop_W <<<(G-1)/threads_per_block + 1, threads_per_block>>> (d_proposed_W, d_W, d_alpha, d_beta, d_nu, d_delta,
                                                                            d_phi, d_X, d_temp_double, i, G);
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[i], G);
        }
        update_W <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_W, d_log_rho, d_pi, d_W, N, K);
        
        //pi
        sample_index = 0;
        for (int b=0; b<B; b++){
            if(b>0)
                sample_index += n_b[b-1];
            for (int k=0; k<K; k++){
                fill_I_W <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>> (&d_W[sample_index], d_temp_int, k, n_b[b]);
                sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (d_temp_int, &d_count[k], n_b[b]);
            }
            //Assuming that K is less than 1024.
            update_pi <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_states, d_count, &d_pi[b*K], K);
        }
    }
    printf("Burnin completed. Start recording iterations.\n");
    //Declare stored iterations.
    double* d_preserved_gamma; cudaMalloc(&d_preserved_gamma, B*2*n_preserved*sizeof(double));
    double* d_preserved_alpha; cudaMalloc(&d_preserved_alpha, G*n_preserved*sizeof(double));
    int* d_preserved_L; cudaMalloc(&d_preserved_L, G*(K-1)*n_preserved*sizeof(int));
    double* d_preserved_p; cudaMalloc(&d_preserved_p, 2*n_preserved*sizeof(double));
    double* d_preserved_beta; cudaMalloc(&d_preserved_beta, G*(K-1)*n_preserved*sizeof(double));
    double* d_preserved_nu; cudaMalloc(&d_preserved_nu, G*(B-1)*n_preserved*sizeof(double));
    double* d_preserved_delta; cudaMalloc(&d_preserved_delta, N*n_preserved*sizeof(double));
    double* d_preserved_phi; cudaMalloc(&d_preserved_phi, B*G*n_preserved*sizeof(double));
    int* d_preserved_W; cudaMalloc(&d_preserved_W, N*n_preserved*sizeof(int));
    double* d_preserved_pi; cudaMalloc(&d_preserved_pi, B*K*n_preserved*sizeof(double));
    
    for(int iter=0; iter<n_preserved; iter++){
        //Z and X
        update_Z_X <<<(N*G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_Y, d_W, d_alpha, d_beta, d_nu,
                        d_delta, d_gamma, d_phi,
                        d_Z, d_X, B, N, G);//If there are not enough threads, may have to separate Z and X.
        
        // gamma
        propose_gamma <<<(B-1)/threads_per_block + 1, threads_per_block>>>(d_states, d_proposed_gamma, d_gamma,B);
        
        //Fill gamma0
        sample_index = 0;
        for (int b=0; b<B; b++){
            if (b>0)
                sample_index += n_b[b-1];
            fill_prop_gamma0 <<<(n_b[b]*G-1)/threads_per_block+1, threads_per_block>>> (d_proposed_gamma, d_gamma, &d_Z[sample_index*G],
                                                                                        &d_X[sample_index*G], d_temp_double, b, B, n_b[b]*G);
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[b], n_b[b]*G);
        }
        update_gamma0 <<< (B-1)/threads_per_block + 1, threads_per_block >>> (d_states, d_proposed_gamma, d_log_rho, d_gamma, B);
        
        //Fill gamma1
        sample_index = 0;
        for (int b=0; b<B; b++){
            if (b>0)
                sample_index += n_b[b-1];
            fill_prop_gamma1 <<<(n_b[b]*G-1)/threads_per_block+1, threads_per_block>>> (d_proposed_gamma, d_gamma, &d_Z[sample_index*G],
                                                                                        &d_X[sample_index*G], d_temp_double, b, B, n_b[b]*G);
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[b], n_b[b]*G);
        }
        update_gamma1 <<< (B-1)/threads_per_block + 1, threads_per_block >>> (d_states, d_proposed_gamma, d_log_rho, d_gamma, B);
        
        //alpha
        propose_alpha <<<(G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_alpha, d_alpha, G);
        
        for (int g=0; g<G; g++){
            fill_prop_alpha <<< (N-1)/threads_per_block + 1, threads_per_block >>>(d_proposed_alpha, d_alpha, d_beta, d_nu, d_delta, d_phi,
                                                                                d_W, d_X, d_temp_double, g, B, N, G);
            sum_on_gpu <double> <<< 1, threads_per_block, threads_per_block*sizeof(double) >>>(d_temp_double, &d_log_rho[g], N);
        }
        update_alpha <<< (G-1)/threads_per_block + 1, threads_per_block >>>(d_states, d_proposed_alpha, d_log_rho, d_mu_alpha, d_alpha, G);
        
        //L
        update_L <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_p, &d_beta[G], &d_L[G], G*(K-1));
        
        //p
        if(iter >= n_unchanged - n_burnin){
            //Fill d_count[0] with sum of d_L and d_p[1] with sum of I(d_L==0)*d_beta^2
            //I used d_p[1] to carry the second sum. Its value will be replaced anyway.
            sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (&d_L[G], &d_count[0], G*(K-1));
            fill_I_L_beta_sq <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (&d_L[G], &d_beta[G], d_temp_double, G*(K-1));
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_p[1], G*(K-1));
            update_p <<<1,1>>> (d_states, d_count, d_p, G, K);
            update_tau0 <<<1,1>>> (d_states, d_count, d_p, G, K);
        }
        
        //beta
        propose_beta <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_beta, &d_beta[G], G*(K-1));
        for (int k=1; k<K; k++){
            for (int g=0; g<G; g++){
                fill_prop_beta <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_proposed_beta, d_beta, d_alpha, d_nu, d_delta,
                                                                                d_phi, d_W, d_X, d_temp_double, g, k, N, G);
                sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>>(d_temp_double, &d_log_rho[(k-1)*G + g], N);
            }
        }
        update_beta <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_beta, d_log_rho, &d_L[G], &d_beta[G], d_p, G*(K-1));
        
        //nu
        propose_nu <<<(G*(B-1)-1)/threads_per_block + 1, threads_per_block>>>(d_states, d_proposed_nu, &d_nu[G], G*(B-1));
        sample_index = 0;
        for (int b=1; b<B; b++){
            sample_index += n_b[b-1];
            for (int g=0; g<G; g++){
                fill_prop_nu <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>>(d_proposed_nu, d_nu, d_alpha, d_beta, &d_delta[sample_index], d_phi, &d_W[sample_index],
                                                                                        &d_X[sample_index*G], d_temp_double, b, g, G, n_b[b]);
                sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[(b-1)*G + g], n_b[b]);
            }
        }
        update_nu <<<(G*(B-1)-1)/threads_per_block+1, threads_per_block>>> (d_states, d_proposed_nu, d_log_rho, &d_nu[G], G, G*(B-1));
        
        //delta
        propose_delta <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_delta, d_delta, N);
        
        for (int i=0; i<N; i++){
            fill_prop_delta <<<(G-1)/threads_per_block + 1, threads_per_block>>> (d_proposed_delta, d_delta, d_alpha, d_beta, d_nu, d_phi,
                                                                                d_W, d_X, d_temp_double, i, G);
            sum_on_gpu <double> <<<(G-1)/threads_per_block + 1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[i], G);
        }
        update_delta <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_delta, d_log_rho, d_mu_delta, d_delta, N);
        
        //phi
        propose_phi <<<(B*G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_phi, d_phi, B*G);
        sample_index = 0;
        int bg = 0;
        for (int b=0; b<B; b++){
            if(b>0)
                sample_index += n_b[b-1];
            for (int g=0; g<G; g++){
                fill_prop_phi <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>> (&d_proposed_phi[bg], &d_phi[bg], &d_alpha[g], &d_beta[g], &d_nu[bg],
                                                                                        &d_delta[sample_index], &d_W[sample_index], &d_X[sample_index*G + g], d_temp_double, G);
                sum_on_gpu <double> <<<1, threads_per_block , threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[bg], n_b[b]);
                bg++;
            }
        }
        update_phi <<<(B*G-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_phi, d_log_rho, d_phi, B*G);
        
        //w
        propose_W <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_W, N, K);
        
        for (int i=0; i<N; i++){
            fill_prop_W <<<(G-1)/threads_per_block + 1, threads_per_block>>> (d_proposed_W, d_W, d_alpha, d_beta, d_nu, d_delta,
                                                                            d_phi, d_X, d_temp_double, i, G);
            sum_on_gpu <double> <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_temp_double, &d_log_rho[i], G);
        }
        update_W <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_states, d_proposed_W, d_log_rho, d_pi, d_W, N, K);
        
        //pi
        sample_index = 0;
        for (int b=0; b<B; b++){
            if(b>0)
                sample_index += n_b[b-1];
            for (int k=0; k<K; k++){
                fill_I_W <<<(n_b[b]-1)/threads_per_block + 1, threads_per_block>>> (&d_W[sample_index], d_temp_int, k, n_b[b]);
                sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (d_temp_int, &d_count[k], n_b[b]);
            }
            //Assuming that K is less than 1024.
            update_pi <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (d_states, d_count, &d_pi[b*K], K);
        }
        //Store iteration.
        store_gamma <<<(B*2-1)/threads_per_block + 1, threads_per_block>>> (d_gamma, d_preserved_gamma, iter, n_preserved, B*2);
        store_alpha <<<(G-1)/threads_per_block + 1, threads_per_block>>> (d_alpha, d_preserved_alpha, iter, n_preserved, G);
        store_L_beta <<<(G*(K-1)-1)/threads_per_block + 1, threads_per_block>>>(&d_L[G], d_preserved_L, &d_beta[G], d_preserved_beta, iter, n_preserved, G*(K-1));
        cudaMemcpy(&d_preserved_p[iter], &d_p[0], sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&d_preserved_p[n_preserved + iter], &d_p[1], sizeof(double), cudaMemcpyDeviceToDevice);
        store_nu_phi <<<(B*G-1)/threads_per_block + 1, threads_per_block>>> (d_nu, d_preserved_nu, d_phi, d_preserved_phi, iter, n_preserved, G, B*G);
        store_delta_W <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_delta, d_preserved_delta, d_W, d_preserved_W, iter, n_preserved, N);
        store_pi <<<(B*K-1)/threads_per_block + 1, threads_per_block>>> (d_pi, d_preserved_pi, iter, n_preserved, B*K);
    }
    
    //3.Posterior Inference
    printf("Start posterior inference.\n");
    //reuse d_{sth} as d_post_{sth}
    //Z
    // Not necessary
    //X
    // Not necessary either
    //gamma
    for (int b=0; b<2*B; b++)
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_gamma[b*n_preserved], &d_gamma[b], n_preserved);
    
    //alpha
    for (int g=0; g<G; g++)
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_alpha[g*n_preserved], &d_alpha[g], n_preserved);
    
    //L
    double* d_post_prob; cudaMalloc(&d_post_prob, G*(K-1)*sizeof(double));
    for (int pos=0; pos<G*(K-1); pos++)
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (&d_preserved_L[pos*n_preserved], &d_post_prob[pos], n_preserved);
    
    //p and tau0
    for (int pos=0; pos<2; pos++)
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_p[pos*n_preserved], &d_p[pos], n_preserved);
    
    //beta
    for (int pos=G; pos<G*K; pos++)//starts with G
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_beta[(pos-G)*n_preserved], &d_beta[pos], n_preserved);
    
    //nu
    for (int pos=G; pos<B*G; pos++)//starts with G
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_nu[(pos-G)*n_preserved], &d_nu[pos], n_preserved); 
    
    //delta
    for (int i=0; i<N; i++)
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_delta[i*n_preserved], &d_delta[i], n_preserved);
    
    //phi
    for (int pos=0; pos<B*G; pos++)
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_phi[pos*n_preserved], &d_phi[pos], n_preserved);
    
    //W
    for (int k=0; k<K; k++){
        for (int i=0; i<N; i++){
            fill_post_W <<<(n_preserved-1)/threads_per_block + 1, threads_per_block>>> (&d_preserved_W[i*n_preserved], d_temp_int, k, n_preserved);
            sum_on_gpu <int> <<<1, threads_per_block, threads_per_block*sizeof(int)>>> (d_temp_int, &d_count[k*N + i], n_preserved);
        }
    }
    mode_on_gpu <<<(N-1)/threads_per_block + 1, threads_per_block>>> (d_count, d_W, N, K);
    
    //pi
    for (int pos=0; pos<B*K; pos++)
        mean_on_gpu <<<1, threads_per_block, threads_per_block*sizeof(double)>>> (&d_preserved_pi[pos*n_preserved], &d_pi[pos], n_preserved);
    
    //4.Write output
    printf("Start writing output files.\n");
    
    //Z
    int* h_post_Z = (int*)malloc(N*G*sizeof(int));
    cudaMemcpy(h_post_Z, d_Z, N*G*sizeof(int), cudaMemcpyDeviceToHost);
    char Z_output_filename[200];
    strcpy(Z_output_filename, output_file);
    strcat(Z_output_filename, "_post_Z.txt");
    FILE *Z_output_file;
    Z_output_file = fopen(Z_output_filename,"w");
    
    for (int i=0; i<N; i++){
        for (int g=0; g<G; g++)
            fprintf(Z_output_file, "%d\t", h_post_Z[i*G + g]);
        fprintf(Z_output_file, "\n");
    }
    fclose(Z_output_file);
    free(h_post_Z);
    
    //X
    int* h_post_X = (int*)malloc(N*G*sizeof(int));
    cudaMemcpy(h_post_X, d_X, N*G*sizeof(int), cudaMemcpyDeviceToHost);
    char X_output_filename[200];
    strcpy(X_output_filename, output_file);
    strcat(X_output_filename, "_post_X.txt");
    FILE *X_output_file;
    X_output_file = fopen(X_output_filename,"w");
    
    for (int i=0; i<N; i++){
        for (int g=0; g<G; g++)
            fprintf(X_output_file, "%d\t", h_post_X[i*G + g]);
        fprintf(X_output_file, "\n");
    }
    fclose(X_output_file);
    free(h_post_X);
    
    //gamma
    double* h_post_gamma = (double*)malloc(2*B*sizeof(double));
    cudaMemcpy(h_post_gamma, d_gamma, 2*B*sizeof(double), cudaMemcpyDeviceToHost);
    char gamma_output_filename[200];
    strcpy(gamma_output_filename, output_file);
    strcat(gamma_output_filename, "_post_gamma.txt");
    FILE *gamma_output_file;
    gamma_output_file = fopen(gamma_output_filename,"w");
    
    for (int i=0; i<2; i++){
        for (int b=0; b<B; b++)
            fprintf(gamma_output_file, "%lf\t", h_post_gamma[i*B + b]);
        fprintf(gamma_output_file, "\n");
    }
    fclose(gamma_output_file);
    free(h_post_gamma);
    
    //alpha
    double* h_post_alpha = (double*)malloc(G*sizeof(double));
    cudaMemcpy(h_post_alpha, d_alpha, G*sizeof(double), cudaMemcpyDeviceToHost);
    char alpha_output_filename[200];
    strcpy(alpha_output_filename, output_file);
    strcat(alpha_output_filename, "_post_alpha.txt");
    FILE *alpha_output_file;
    alpha_output_file = fopen(alpha_output_filename,"w");
    
    for (int g=0; g<G; g++){
        fprintf(alpha_output_file, "%lf\n", h_post_alpha[g]);
    }
    fclose(alpha_output_file);
    free(h_post_alpha);
    
    //L ('s post prob)
    double* h_post_L = (double*)malloc(G*(K-1)*sizeof(double));
    cudaMemcpy(h_post_L, d_post_prob, G*(K-1)*sizeof(double), cudaMemcpyDeviceToHost);
    char L_output_filename[200];
    strcpy(L_output_filename, output_file);
    strcat(L_output_filename, "_post_L.txt");
    FILE *L_output_file;
    L_output_file = fopen(L_output_filename,"w");
    
    for (int k=1; k<K; k++){
        for (int g=0; g<G; g++)
            fprintf(L_output_file, "%lf\t", h_post_L[(k-1)*G + g]);
        fprintf(L_output_file, "\n");
    }
    fclose(L_output_file);
    free(h_post_L);
    
    //beta
    double* h_post_beta = (double*)malloc(G*(K-1)*sizeof(double));
    cudaMemcpy(h_post_beta, &d_beta[G], G*(K-1)*sizeof(double), cudaMemcpyDeviceToHost);
    char beta_output_filename[200];
    strcpy(beta_output_filename, output_file);
    strcat(beta_output_filename, "_post_beta.txt");
    FILE *beta_output_file;
    beta_output_file = fopen(beta_output_filename,"w");
    
    for (int g=0; g<G; g++)
        fprintf(beta_output_file, "0.000000\t");//Directly print 0 to save time
    fprintf(beta_output_file, "\n");
    for (int k=1; k<K; k++){
        for (int g=0; g<G; g++)
            fprintf(beta_output_file, "%lf\t", h_post_beta[(k-1)*G + g]);
        fprintf(beta_output_file, "\n");
    }
    fclose(beta_output_file);
    free(h_post_beta);
    
    //p and tau0
    double* h_post_p = (double*)malloc(2*sizeof(double));
    cudaMemcpy(h_post_p, d_p, 2*sizeof(double), cudaMemcpyDeviceToHost);
    char p_output_filename[200];
    strcpy(p_output_filename, output_file);
    strcat(p_output_filename, "_post_p.txt");
    FILE *p_output_file;
    p_output_file = fopen(p_output_filename,"w");
    
    for (int i=0; i<2; i++){
        fprintf(p_output_file, "%lf\t", h_post_p[i]);
    }
    fclose(p_output_file);
    free(h_post_p);
    
    //nu
    double* h_post_nu = (double*)malloc(G*(B-1)*sizeof(double));
    cudaMemcpy(h_post_nu, &d_nu[G], G*(B-1)*sizeof(double), cudaMemcpyDeviceToHost);
    char nu_output_filename[200];
    strcpy(nu_output_filename, output_file);
    strcat(nu_output_filename, "_post_nu.txt");
    FILE *nu_output_file;
    nu_output_file = fopen(nu_output_filename,"w");
    
    for (int g=0; g<G; g++)
        fprintf(nu_output_file, "0.000000\t");//Directly print 0 to save time
    fprintf(nu_output_file, "\n");
    for (int b=1; b<B; b++){
        for (int g=0; g<G; g++)
            fprintf(nu_output_file, "%lf\t", h_post_nu[(b-1)*G + g]);
        fprintf(nu_output_file, "\n");
    }
    fclose(nu_output_file);
    free(h_post_nu);
    
    //delta
    double* h_post_delta = (double*)malloc(N*sizeof(double));
    cudaMemcpy(h_post_delta, d_delta, N*sizeof(double), cudaMemcpyDeviceToHost);
    char delta_output_filename[200];
    strcpy(delta_output_filename, output_file);
    strcat(delta_output_filename, "_post_delta.txt");
    FILE *delta_output_file;
    delta_output_file = fopen(delta_output_filename,"w");

    for (int i=0; i<N; i++){
            fprintf(delta_output_file, "%lf\n", h_post_delta[i]);
    }
    fclose(delta_output_file);
    free(h_post_delta);
    
    //phi
    double* h_post_phi = (double*)malloc(G*B*sizeof(double));
    cudaMemcpy(h_post_phi, d_phi, G*B*sizeof(double), cudaMemcpyDeviceToHost);
    char phi_output_filename[200];
    strcpy(phi_output_filename, output_file);
    strcat(phi_output_filename, "_post_phi.txt");
    FILE *phi_output_file;
    phi_output_file = fopen(phi_output_filename,"w");
    
    for (int b=0; b<B; b++){
        for (int g=0; g<G; g++)
            fprintf(phi_output_file, "%lf\t", h_post_phi[b*G + g]);
        fprintf(phi_output_file, "\n");
    }
    fclose(phi_output_file);
    free(h_post_phi);
    
    //W
    int* h_post_W = (int*)malloc(N*sizeof(int));
    cudaMemcpy(h_post_W, d_W, N*sizeof(int), cudaMemcpyDeviceToHost);
    char W_output_filename[200];
    strcpy(W_output_filename, output_file);
    strcat(W_output_filename, "_post_W.txt");
    FILE *W_output_file;
    W_output_file = fopen(W_output_filename,"w");

    for (int i=0; i<N; i++){
            fprintf(W_output_file, "%d\n", h_post_W[i]);
    }
    fclose(W_output_file);
    free(h_post_W);
    
    //pi
    double* h_post_pi = (double*)malloc(B*K*sizeof(double));
    cudaMemcpy(h_post_pi, d_pi, B*K*sizeof(double), cudaMemcpyDeviceToHost);
    char pi_output_filename[200];
    strcpy(pi_output_filename, output_file);
    strcat(pi_output_filename, "_post_pi.txt");
    FILE *pi_output_file;
    pi_output_file = fopen(pi_output_filename,"w");
    
    for (int b=0; b<B; b++){
        for (int k=0; k<K; k++)
            fprintf(pi_output_file, "%lf\t", h_post_pi[b*K + k]);
        fprintf(pi_output_file, "\n");
    }
    fclose(pi_output_file);
    free(h_post_pi);
    
    //(Optional) Print out all preserved iterations.
    if (print_preserved){
        printf("Start writing preserved iterations.\n");
        
        //gamma
        double* h_preserved_gamma = (double*)malloc(2*B*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_gamma, d_preserved_gamma, 2*B*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char gamma_preserved_filename[200];
        strcpy(gamma_preserved_filename, output_file);
        strcat(gamma_preserved_filename, "_preserved_gamma.txt");
        FILE *gamma_preserved_file;
        gamma_preserved_file = fopen(gamma_preserved_filename,"w");
        
        for (int pos=0; pos<2*B; pos++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(gamma_preserved_file, "%lf\t", h_preserved_gamma[pos*n_preserved + iter]);
            fprintf(gamma_preserved_file, "\n");
        }
        fclose(gamma_preserved_file);
        free(h_preserved_gamma);
        
        //alpha
        double* h_preserved_alpha = (double*)malloc(G*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_alpha, d_preserved_alpha, G*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char alpha_preserved_filename[200];
        strcpy(alpha_preserved_filename, output_file);
        strcat(alpha_preserved_filename, "_preserved_alpha.txt");
        FILE *alpha_preserved_file;
        alpha_preserved_file = fopen(alpha_preserved_filename,"w");
        
        for (int g=0; g<G; g++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(alpha_preserved_file, "%lf\t", h_preserved_alpha[g]);
            fprintf(alpha_preserved_file, "\n");
        }
        fclose(alpha_preserved_file);
        free(h_preserved_alpha);
        
        //L
        int* h_preserved_L = (int*)malloc(G*(K-1)*n_preserved*sizeof(int));
        cudaMemcpy(h_preserved_L, d_preserved_L, G*(K-1)*n_preserved*sizeof(int), cudaMemcpyDeviceToHost);
        char L_preserved_filename[200];
        strcpy(L_preserved_filename, output_file);
        strcat(L_preserved_filename, "_preserved_L.txt");
        FILE *L_preserved_file;
        L_preserved_file = fopen(L_preserved_filename,"w");
        
        for (int pos=0; pos<(K-1)*G; pos++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(L_preserved_file, "%d\t", h_preserved_L[pos*n_preserved + iter]);
            fprintf(L_preserved_file, "\n");
        }
        fclose(L_preserved_file);
        free(h_preserved_L);
        
        //beta
        double* h_preserved_beta = (double*)malloc(G*(K-1)*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_beta, d_preserved_beta, G*(K-1)*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char beta_preserved_filename[200];
        strcpy(beta_preserved_filename, output_file);
        strcat(beta_preserved_filename, "_preserved_beta.txt");
        FILE *beta_preserved_file;
        beta_preserved_file = fopen(beta_preserved_filename,"w");
        
        for (int pos=0; pos<G*(K-1); pos++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(beta_preserved_file, "%lf\t", h_preserved_beta[pos*n_preserved + iter]);
            fprintf(beta_preserved_file, "\n");
        }
        fclose(beta_preserved_file);
        free(h_preserved_beta);
        
        //p and tau0
        double* h_preserved_p = (double*)malloc(2*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_p, d_preserved_p, 2*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char p_preserved_filename[200];
        strcpy(p_preserved_filename, output_file);
        strcat(p_preserved_filename, "_preserved_p.txt");
        FILE *p_preserved_file;
        p_preserved_file = fopen(p_preserved_filename,"w");
        
        for (int i=0; i<2; i++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(p_preserved_file, "%lf\t", h_preserved_p[i*n_preserved + iter]);
            fprintf(p_preserved_file, "\n");
        }
        fclose(p_preserved_file);
        free(h_preserved_p);
        
        //nu
        double* h_preserved_nu = (double*)malloc(G*(B-1)*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_nu, d_preserved_nu, G*(B-1)*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char nu_preserved_filename[200];
        strcpy(nu_preserved_filename, output_file);
        strcat(nu_preserved_filename, "_preserved_nu.txt");
        FILE *nu_preserved_file;
        nu_preserved_file = fopen(nu_preserved_filename,"w");
        
        for (int pos=0; pos<(B-1)*G; pos++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(nu_preserved_file, "%lf\t", h_preserved_nu[pos*n_preserved + iter]);
            fprintf(nu_preserved_file, "\n");
        }
        fclose(nu_preserved_file);
        free(h_preserved_nu);
        
        //delta
        double* h_preserved_delta = (double*)malloc(N*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_delta, d_preserved_delta, N*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char delta_preserved_filename[200];
        strcpy(delta_preserved_filename, output_file);
        strcat(delta_preserved_filename, "_preserved_delta.txt");
        FILE *delta_preserved_file;
        delta_preserved_file = fopen(delta_preserved_filename,"w");

        for (int i=0; i<N; i++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(delta_preserved_file, "%lf\t", h_preserved_delta[i*n_preserved + iter]);
            fprintf(delta_preserved_file, "\n");
        }
        fclose(delta_preserved_file);
        free(h_preserved_delta);
        
        //phi
        double* h_preserved_phi = (double*)malloc(G*B*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_phi, d_preserved_phi, G*B*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char phi_preserved_filename[200];
        strcpy(phi_preserved_filename, output_file);
        strcat(phi_preserved_filename, "_preserved_phi.txt");
        FILE *phi_preserved_file;
        phi_preserved_file = fopen(phi_preserved_filename,"w");
        
        for (int pos=0; pos<B*G; pos++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(phi_preserved_file, "%lf\t", h_preserved_phi[pos*n_preserved + iter]);
            fprintf(phi_preserved_file, "\n");
        }
        fclose(phi_preserved_file);
        free(h_preserved_phi);
        
        //W
        int* h_preserved_W = (int*)malloc(N*n_preserved*sizeof(int));
        cudaMemcpy(h_preserved_W, d_preserved_W, N*n_preserved*sizeof(int), cudaMemcpyDeviceToHost);
        char W_preserved_filename[200];
        strcpy(W_preserved_filename, output_file);
        strcat(W_preserved_filename, "_preserved_W.txt");
        FILE *W_preserved_file;
        W_preserved_file = fopen(W_preserved_filename,"w");

        for (int i=0; i<N; i++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(W_preserved_file, "%d\t", h_preserved_W[i*n_preserved + iter]);
            fprintf(W_preserved_file, "\n");
        }
        fclose(W_preserved_file);
        free(h_preserved_W);
        
        //pi
        double* h_preserved_pi = (double*)malloc(B*K*n_preserved*sizeof(double));
        cudaMemcpy(h_preserved_pi, d_preserved_pi, B*K*n_preserved*sizeof(double), cudaMemcpyDeviceToHost);
        char pi_preserved_filename[200];
        strcpy(pi_preserved_filename, output_file);
        strcat(pi_preserved_filename, "_preserved_pi.txt");
        FILE *pi_preserved_file;
        pi_preserved_file = fopen(pi_preserved_filename,"w");
        
        for (int pos=0; pos<B*K; pos++){
            for (int iter=0; iter<n_preserved; iter++)
                fprintf(pi_preserved_file, "%lf\t", h_preserved_pi[pos*n_preserved + iter]);
            fprintf(pi_preserved_file, "\n");
        }
        fclose(pi_preserved_file);
        free(h_preserved_pi);
    }
    
    return (0);
}

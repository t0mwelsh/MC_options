/* Program that uses Monte Carlo simulations to estimate the price of a floating-strike Asian put option on a pre-defined stock. 
Firstly, the value of the option is estimated taking random draws from a standard normal distribution to propagate the stock
along a path for the price to be calculated. The results for increasing numbers of paths are then outputted to a file. Secondly,
this is done with 100 repeats to give an average and then an error on this average for increasing number of paths. In this case,
the random draws are moment-matched to improve the accuracy of the results.

The put option is on a stock price which follows a risk-neutral distribution stochastic process with SDE
dS = f(S, t)dt + v(S, t)dW, where W is a Wiener process and f and v are calibrated functions: f(S, t) = αθ − βS, 
v(S, t) = σ(|S|)^γ dW. The put option was path dependentat and so a Euler type scheme was used to get from one point to the next:
S(t_k) = S(t_{k−1}) + f(S(t_{k−1}), t_{k−1})∆t + v(S(t_{k−1}), t_{k−1})√{∆t}ϕ_{i,k−1} for t_k the k-th time-step. In this problem,
we had parameters α=0.02, β = 0.02, θ = 70, γ = 1.03, T = 0.5, the time at option maturity in years, σ = 0.19, the volatility. */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>

std::vector<double> mean_and_error(const std::vector<double>& samples)
// calculates the mean and the error on the mean from a sample
{
    double sum = 0.;
    size_t size = samples.size();
    for (const auto& sample : samples)
    {
        sum += sample;
    }
    double mean = sum / size;
    // estimate the variance from the sample
    double sumvar = 0.;
    for (int i = 0; i < size; i++)
    {
        sumvar += (samples[i] - mean) * (samples[i] - mean);
    }
    double variance = (size > 1) ? sumvar / (size - 1) : 0.0;

    // get the standard deviation of the sample mean
    double sd = sqrt(variance / size);

    std::vector<double> mean_and_error = { mean, sd };
    return mean_and_error;
}

double f_func(std::vector<double> params, double stock_price)
// pre-defined function
{
    return params[4] * params[6] - params[5] * stock_price;
}

double v_func(std::vector<double> params, double stock_price)
// pre-defined function
{
    return params[7] * pow(abs(stock_price), params[8]);
}

double stock_propagator(double stock_price, double dt, double phi, std::vector<double> params)
// Euler type scheme to propagate the value of the stock along a path
{
    return stock_price + f_func(params, stock_price) * dt + v_func(params, stock_price) * sqrt(dt) * phi;
}

double mc_path(const std::vector<double>& params, int K, int N)
// Monte-Carlo simulation that takes random draws from a standard normal distribution to estimate the
// value of a floating-strike Asian put option on a pre-defined stock
{
    static std::mt19937 rng;
    std::normal_distribution<> ND(0., 1.);
    double sum = 0.;
    double dt = params[3] / K;
    for (int n = 0; n < N; n++)
    {
        // now create a path
        std::vector<double> stock_path(K + 1);
        stock_path[0] = params[0];
        for (int i = 1; i <= K; i++)
        {
            double phi = ND(rng);
            stock_path[i] = stock_propagator(stock_path[i - 1], dt, phi, params);
        }
        // and calculate A the value of the Asian option
        double A = 0.;
        for (const auto& stock_price : stock_path)
        {
            A += stock_price;
        }
        A /= K;
        // add in the payoff to the sum
        sum = sum + std::max(A - stock_path[K], 0.);
    }
    return sum / N * exp(-params[2] * params[3]); // multiply for discounted payoff due to risk-free interest rate
}

std::vector<double> moment_match_vector_func(int N) 
// creates a vector of random draws from a standard normal distribution with a mean of 0 and a variance of 1
{
    std::vector<double> phi_vector(2 * N);
    static std::mt19937 rng;
    std::normal_distribution<> ND(0., 1.);
    double phi;
    double sum{ 0 };

    for (int i = 0; i < N; i++) {
        phi = ND(rng);
        phi_vector[2 * i] = phi;
        phi_vector[2 * i + 1] = -phi; // means we get an average of 0 as desired in a normal distribution
        sum += 2 * phi * phi;
    }

    sum /= (2 * N - 1);
    double std_dev{ sqrt(sum) };
    for (int i = 0; i < 2 * N; i++) {
        phi_vector[i] /= std_dev; // means we get a variance (and standard deviation) of 1 as desired 
    }
    return phi_vector;
}

std::vector<std::vector<double>> repeated_mc_path_moment_match(std::vector<double> params, int repeats, int K, int max_N, int step_size)
// creates an array of estimated values of a floating-strike Asian put option on a pre-defined stock using Monte Carlo simulations with
// increasing numbers of paths and a certain number of repeats. Random draws from a standard normal distribution are moment-matched. 
{
    double dt = params[3] / K;
    std::vector<double> sample(max_N);
    std::vector<double> Ns(max_N / step_size);
    std::vector<std::vector<double>> big_boi(repeats+1, std::vector<double>(max_N / step_size));

    for (int i = 0; i < max_N / step_size; i++)
    {
        Ns[i] = step_size * (i+1);
    }
    big_boi[0] = Ns;

    // now instantiate a path
    std::vector<double> stock_path(K + 1);
    stock_path[0] = params[0];

    for (int M = 1; M <= repeats; M++)
    {
        std::vector<double> phi_vector = moment_match_vector_func(max_N * K / 2); // moment_match_vector_func creates max_N*K phi values
        for (int n = 0; n < max_N; n++)
        {
            for (int i = 1; i <= K; i++)
            {
                double phi = phi_vector[(i - 1) * max_N + n]; //phi vector is made from moment-matched variables and need each draw of a path
                // to be independent so can't take consecutive ones as otherwise we'd get phi and -phi
                stock_path[i] = stock_propagator(stock_path[i - 1], dt, phi, params); //overwrite stock_path each time
            }
            // and calculate A
            double A = 0.;
            for (int i = 1; i <= K; i++)
            {
                A += stock_path[i];
            }
            A /= K;
            // add in the payoff to the sum
            sample[n] = std::max(A - stock_path[K], 0.);
        }

        double sum = 0;
        for (int n = step_size; n <= max_N; n += step_size)
        { // we use just one set of max_N paths for each repeat and then take successively increasing
          // number of paths for a specific n. This greatly increases efficiency (instead of creating
          // 500 paths scrapping those and then 1000 paths and so on, for example) and was not found
          // to statiscally impair the results
            for (int i = n - step_size; i < n; i++)
            {
                sum += sample[i];
            }
            Ns[n / step_size - 1] = sum / n * exp(-params[2] * params[3]); // multiply for discounted payoff due to risk-free interest rate
        }

        big_boi[M] = Ns; // Mth row of big_boi is then the estimated value for increasing values of N paths 
    }
    return big_boi;
}

void output_averages_to_file(std::vector<std::vector<double>> big_boi, std::string file_name)
// takes an array of estimated values with increasing numbers of paths and repeats and outputs to a file the averages and error
// on each average for the estimated values for each number of paths
{
    int max_N = big_boi[0].back();
    int step_size = big_boi[0][0];
    int repeats = big_boi.size() - 1;
    big_boi.erase(big_boi.begin()); //don't need this anymore as we have variables above and useful to remove for calculations
    
    std::vector<std::vector<double>> result(max_N / step_size, std::vector<double>(2));
    std::vector<double> column(repeats);
    for (int i = 0; i < max_N / step_size; i++)
    {
        for (int j = 0; j < repeats; j++) {
            column[j] = big_boi[j][i]; // a column is the estimated value for a specific N for each repeat
        }
        result[i] = mean_and_error(column);
    }

    std::ofstream myfile;
    myfile.open(file_name);
    myfile << "N" << "," << "value" << "," << "error" << "," << "\n";
    for (int i = 0; i < result.size(); i++) {
        myfile << step_size * (i + 1) << "," << result[i][0] << "," << result[i][1] << "," << "\n";
    }
    myfile.close();
}

void output_value_to_file(std::vector<double> params, int K, std::string file_name, int step_size, int max_N) 
// outputs to file the estimated values for increasing numbers of paths of a floating-strike Asian put option on a pre-defined stock
{
    std::ofstream myfile;
    myfile.open(file_name);

    std::vector<double> phi_vector;
    for (int i = 1; i <= max_N / step_size; i++) {
        myfile << step_size * i << "," << mc_path(params, K, i * step_size) << "\n";
    }
    myfile.close();
}

int main()
{
    double initial_stock_price = 70.0231, strike_price = 70., interest_rate = 0.01, maturity_time = 0.5,
        alpha = 0.02, beta = 0.02, theta = 70., volatility = 0.19, gamma = 1.03;
    std::vector<double> params{ initial_stock_price, strike_price, interest_rate, maturity_time, alpha, beta, theta, volatility, gamma };

    int K = 70; // number of points on the path
    
    output_value_to_file(params, K, "mc_path_data.csv", 1000, 50000);

    std::vector<std::vector<double>> big_boi = repeated_mc_path_moment_match(params, 100, K, 50000, 1000);
    output_averages_to_file(big_boi, "moment_match_path_dep_average_data.csv");
    
    return 0;
}
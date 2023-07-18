/* Program that uses Monte Carlo simulations to estimate the price of a European put option. Random
draws from a standard normal distribution are generated using Halton sequences and a Box-Muller
technique.

The put option is on a stock price which follows a risk-neutral distribution at time t as 
S_t ∼ N(f(S_0, t), v^2(S_0, t)t), where S_0 is the current stock price, S_t is the stock price at t; 
f and v are calibrated functions. Here, f(S0, t) = S0(αT + tan(βT)) + θcos(αT + βT),
v(S_0, t) = 1/2 σ(1 + αT)(S_0 + θ)^γ, where we have parameters α=0.02, β = 0.02, θ = 70, γ = 1.03, T = 1, 
the time at option maturity in years, σ = 0.19, the volatility. The strike price is set at 70. */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

double monte_carlo(std::vector<double> params, std::vector<double> phi_vector)
// Monte Carlo simulation that gives an approximation to the specific European option
{
	double sum = 0.; // initialise sum
	int N = phi_vector.size();
	double new_stock_price;
	for (int i = 0; i < N; i++) // iterate through phi vector
	{
		double phi = phi_vector[i];
		new_stock_price = params[0] * (params[4] * params[3] + tan(params[5] * params[3])) + params[6] * 
			cos(params[4] * params[3] + params[5] * params[3]) + params[7] * (1 + params[4] * params[3]) * 0.5 
			* pow(params[0] + params[6], params[8]) * pow(params[3], 0.5) * phi;
		sum = sum + std::max(params[1] - new_stock_price, 0.); // european put option
	}
	return sum / N * exp(-params[2] * params[3]); // multiply for discounted payoff due to risk-free interest rate
}

double Halton_Seq(int index, int base)
// returns the index-th member of the Halton sequence with chosen base 
{
	double f = 1, r = 0;
	while (index > 0) {
		f = f / base;
		r = r + f * (index % base);
		index = index / base;
	}
	return r;
}

std::vector<std::vector<double>> Halton_vector_func(int a, int b, int N) {
	// creates a 2xN array of pairs of numbers where the m-th pair corresponds to the m-th members
	// of Halton sequences with base a and b respectively
	std::vector<std::vector<double>> Halton_vector(N, std::vector<double>(2));
	for (int i = 0; i < N; i++) {
		Halton_vector[i][0] = Halton_Seq(i + 1, a);
		Halton_vector[i][1] = Halton_Seq(i + 1, b);
	}
	return Halton_vector;
}

std::vector<double> box_muller_func(const std::vector<std::vector<double>>& Halton_vector) {
	// creates standard normally distributed numbers from random coordinates on a unit square
	std::vector<double> box_muller_vec(2 * Halton_vector.size());
	for (int i = 0; i < Halton_vector.size(); i++) {
		double x1 = Halton_vector[i][0]; double x2 = Halton_vector[i][1];

		box_muller_vec[2 * i] = cos(2 * atan(1) * 4 * x2) * sqrt(-2 * log(x1));
		box_muller_vec[2 * i + 1] = sin(2 * atan(1) * 4 * x1) * sqrt(-2 * log(x2));
	}
	return box_muller_vec;
}

std::vector<double> Halton_phi_vector_func(int p1, int p2, int N)
// creates standard normally distributed vector of numbers of size 2N
{
	return box_muller_func(Halton_vector_func(p1, p2, N));
}

void output_to_file_vector(const std::vector<double>& params, std::string file_name, int step_size, int max_N) 
// outputs to a file an array of Monte Carlo approximations for increasing number of paths
{
	std::ofstream myfile;
	myfile.open(file_name);
	for (int i = 1; i <= max_N / step_size; i++) {
		std::vector<double> phi_vector = Halton_phi_vector_func(2, 3, step_size * i / 2);
		myfile << step_size * i << "," << monte_carlo(params,
			phi_vector) << "\n";
	}
	myfile.close();
}

int main()
{
	double initial_stock_price = 70.0231, strike_price = 70., interest_rate = 0.01, maturity_time = 1.,
		alpha = 0.02, beta = 0.02, theta = 70., volatility = 0.19, gamma = 1.03;
	std::vector<double> params{ initial_stock_price, strike_price, interest_rate, maturity_time, alpha, beta, theta, volatility, gamma };
	
	output_to_file_vector(params, "halton_sequence_monte_carlo_data.csv", 500, 50000);
	
	return 0;
}
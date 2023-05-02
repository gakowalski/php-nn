<?php

require 'vendor/autoload.php';

use SciPhp\NumPhp as np;

function random_sample(int $size) : array {
    // create an array of random numbers between 0 and 1 of size $size
    // numbers must have standard normal distribution
    // https://en.wikipedia.org/wiki/Normal_distribution
    $random_numbers = array();
    for ($i = 0; $i < $size; $i++) {
        $random_numbers[] = rand(0, 1);
    }
    return $random_numbers;
}

function random_sample_2d(int $size_1, int $size_2) {
    // create an array of random numbers between 0 and 1 of size $size_1 * $size_2
    // numbers must have standard normal distribution
    // https://en.wikipedia.org/wiki/Normal_distribution
    $random_numbers = array();
    for ($i = 0; $i < $size_1; $i++) {
        $random_numbers[] = random_sample($size_2);
    }
    return $random_numbers;
}

function array_fill_2d(int $size_1, int $size_2, $value) {
    // create an array of zeros of size $size_1 * $size_2
    $zeros = array();
    for ($i = 0; $i < $size_1; $i++) {
        $zeros[] = array_fill(0, $size_2, $value);
    }
    return $zeros;
}

function array_fill_recursive(array $array, $value) {
    // fill an array with a value recursively
    foreach ($array as $key => $val) {
        if (is_array($val)) {
            $array[$key] = array_fill_recursive($val, $value);
        } else {
            $array[$key] = $value;
        }
    }
    return $array;
}

// recursively multiply an array by a value
// array is being passed by reference
function multiply_array_by_value(array &$matrix, float $value) {
    foreach ($matrix as $key => $val) {
        if (is_array($val)) {
            multiply_array_by_value($matrix[$key], $value);
        } else {
            $matrix[$key] *= $value;
        }
    }
    return $matrix;
}

function dot_product(array $a, array $b) {
    // dot product of two arrays
    // https://en.wikipedia.org/wiki/Dot_product
    return trader_mult($a, $b); //< requires trader.dll
}

function transpose(array $a) : array {
    // transpose of a matrix
    // https://en.wikipedia.org/wiki/Transpose
    if (count($a) < 2) return $a;
    return array_map(null, ...$a);
}

function matrix_multiply(array $a, array $b) {
    // matrix multiplication
    // https://en.wikipedia.org/wiki/Matrix_multiplication
    $b = transpose($b);
    $result = array();
    for ($i = 0; $i < count($a); $i++) {
        for ($j = 0; $j < count($b); $j++) {
            
        }
    }
    return $result;
}

function sigmoid(float $value) : float {
    // sigmoid function
    // https://en.wikipedia.org/wiki/Sigmoid_function
    return 1.0 / (1.0 + exp(-$value));
}

function sigmoid_array(array $array) {
    // derivative of the sigmoid function
    // https://en.wikipedia.org/wiki/Sigmoid_function
    return array_map('sigmoid', $array);
}

function sigmoid_prime(float $value) : float {
    // derivative of the sigmoid function
    // https://en.wikipedia.org/wiki/Sigmoid_function
    return sigmoid($value) * (1 - sigmoid($value));
}

function sigmoid_prime_array(array $array) {
    // derivative of the sigmoid function
    // https://en.wikipedia.org/wiki/Sigmoid_function
    return array_map('sigmoid_prime', $array);
}

class Network {

    public $layer_sizes = array();
    public $biases = array();
    public $weights = array();

    public function __construct(array $layer_sizes) {
        echo "Network class loaded";

        $this->layer_sizes = $layer_sizes;

        // create biases
        for ($i = 1; $i < count($layer_sizes); $i++) {
            //$this->biases[] = random_sample($layer_sizes[$i]);
            $this->biases[] = np::random()->randn($layer_sizes[$i]);
        }

        // create weights
        for ($i = 0; $i < count($layer_sizes) - 1; $i++) {
            $this->weights[] = random_sample_2d($layer_sizes[$i], $layer_sizes[$i + 1]);
        }
    }

    public function feedforward($value) {
        // feedforward algorithm
        // https://en.wikipedia.org/wiki/Feedforward_neural_network
        for ($i = 0; $i < count($this->layer_sizes) - 1; $i++) {
            $value = sigmoid(array_sum(array_map(function($x, $y) { return $x * $y; }, $this->weights[$i], $value)) + $this->biases[$i]);
        }
        return $value;
    }

    public function stochastic_gradient_descent(array $training_data, int $epochs, int $mini_batch_size, float $learning_rate, array $test_data = null) {
        if ($test_data) {
            $n_test = count($test_data);
        }
        $n = count($training_data);
        for ($j = 0; $j < $epochs; $j++) {
            shuffle($training_data);
            $mini_batches = array();
            for ($k = 0; $k < $n; $k += $mini_batch_size) {
                $mini_batches[] = array_slice($training_data, $k, $mini_batch_size);
            }
            foreach ($mini_batches as $mini_batch) {
                $this->update_mini_batch($mini_batch, $learning_rate);
            }
            if ($test_data) {
                echo "Epoch $j: " . $this->evaluate($test_data) . " / $n_test\n";
            } else {
                echo "Epoch $j complete\n";
            }
        }
    }

    public function update_mini_batch(array $mini_batch, float $learning_rate) {
        $nabla_b = array_fill_recursive($this->biases, 0.0);
        $nabla_w = array_fill_recursive($this->weights, 0.0);
        foreach ($mini_batch as $tuple) {
            list($x, $y) = $tuple;
            list($delta_nabla_b, $delta_nabla_w) = $this->backpropagate($x, $y);
            $nabla_b = array_map(function($x, $y) { return $x + $y; }, $nabla_b, $delta_nabla_b);
            $nabla_w = array_map(function($x, $y) { return $x + $y; }, $nabla_w, $delta_nabla_w);
        }
        $this->weights = array_map(function($x, $y) use ($learning_rate, $mini_batch) { return $x - ($learning_rate / count($mini_batch)) * $y; }, $this->weights, $nabla_w);
        $this->biases = array_map(function($x, $y) use ($learning_rate, $mini_batch) { return $x - ($learning_rate / count($mini_batch)) * $y; }, $this->biases, $nabla_b);
    }

    public function backpropagate($activation, $y) : array {
        $nabla_b = array_fill_recursive($this->biases, 0.0);
        $nabla_w = array_fill_recursive($this->weights, 0.0);

        // feedforward
        $activations = [$activation];
        $zs = array();
        for ($i = 0; $i < count($this->layer_sizes) - 1; $i++) {
            $z = trader_add(dot_product($this->weights[$i], $activation), $this->biases[$i]);
            $zs[] = $z;
            $activation = array_map('sigmoid', $z);
            $activations[] = $activation;
        }

        // backward pass
        $delta = dot_product($this->cost_derivative(array_slice($activations, -1, 1), $y), array_map('sigmoid_prime_array', array_slice($zs, -1, 1)));
        $nabla_b[count($nabla_b) - 1] = $delta;
        $nabla_w[count($nabla_w) - 1] = array_map(function($x, $y) { return $x * $y; }, $delta, $activations[count($activations) - 2]);
        for ($l = 2; $l < count($this->layer_sizes); $l++) {
            $z = $zs[count($zs) - $l];
            $sp = sigmoid_prime_array($z);
            $delta = matrix_multiply(dot_product(transpose(array_slice($this->weights, 1 - $l, 1)), $delta), $sp);
            $nabla_b[count($nabla_b) - $l] = $delta;
            $nabla_w[count($nabla_w) - $l] = array_map(function($x, $y) { return $x * $y; }, $delta, $activations[count($activations) - $l - 1]);
        }
        return array($nabla_b, $nabla_w);
    }

    public function evaluate(array $test_data) : int {
        $test_results = array_map(function($x) { return array_search(max($x), $x); }, array_map(function($x) { return $this->feedforward($x[0]); }, $test_data));
        return array_sum(array_map(function($x, $y) { return $x == $y ? 1 : 0; }, $test_results, array_map(function($x) { return array_search(max($x), $x); }, array_map(function($x) { return $x[1]; }, $test_data))));
    }

    public function cost_derivative(array $output_activations, $y) : array {
        //return array_map(function($x, $y) { return $x - $y; }, $output_activations, $y);
        return trader_sub($output_activations, $y);
    }

    public function num_of_layers() : int {
        return count($this->layer_sizes);
    }

    public function num_of_bias_layers() : int {
        return count($this->layer_sizes) - 1;
    }

    public function num_of_hidden_layers() : int {
        return count($this->layer_sizes) - 2;
    }
}

$network = new Network([2, 3, 1]);



$training_data = array(
    array(array(0, 0), array(0)),
    array(array(0, 1), array(1)),
    array(array(1, 0), array(1)),
    array(array(1, 1), array(0)),
);

//$network->stochastic_gradient_descent($training_data, 1000, 4, 3.0, $training_data);

var_dump(matrix_multiply(
        [[3,7], [4,9]], 
        [[6,2], [5,8]]
    )
);

//echo $network->feedforward(array(0, 0))[0] . "\n";
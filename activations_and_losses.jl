# Vectorized activation functions and their derivatives
# These functions all operate on matrices of size layer_nodes X batch_size.

function linear_activation(input_batch)
    return input_batch 
end

function linear_derivative(activation_batch)
    return 1 
end

function sigmoid_activation(input_batch)
    return 1 ./(1 .+ exp.(-input_batch))
end

function sigmoid_derivative(activation_batch)
    return activation_batch .* (1 .- activation_batch)
end

function tanh_activation(input_batch)
    (1 .- exp.(-2 .* input_batch))./( 1 .+ exp.(-2 .* input_batch))
end

function tanh_derivative(activation_batch)
     1 .-(activation_batch).^2
end

function ReLU_activation(input_batch)
    max.(0,input_batch)
end

function ReLU_derivative(activation_batch) 
    RelU_activation_matrix = copy(activation_batch)
    RelU_activation_matrix[RelU_activation_matrix .>= 0] .= 1
    RelU_activation_matrix[RelU_activation_matrix .< 0] .= 0       
    return RelU_activation_matrix
end

function softmax_activation(input_batch)
    exp.(input_batch)./sum(exp.(input_batch), dims=1)
end

# This function is here only to play nice with the DenseNetwork constructors.
# You don't need to implement it!
function softmax_derivative(activation_batch)
    error("This function should never get called.")
end

# Vectorized loss functions
# Each loss function outputs a single number: the mean loss over the batch

# SSE loss for a data point is a sum over  of squared errors over the output dimensions
function SSE_loss(outputs::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real})
    n = size(outputs,1)
    (1 ./n) .* sum((outputs .- targets).^2)
end

# what do the outputs and targets mean?
function CCE_loss(outputs::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real})
    n = size(outputs,1)
   - sum((targets.*log.(outputs)))/n
end

# example usage: get_derivative(ReLU_activation) returns ReLU_derivative
function get_derivative(activation_function::Function)
    function_string = String(Symbol(activation_function))
    if endswith(function_string, "_activation")
        return getfield(Main, Symbol(function_string[1:end-10] * "derivative"))
    else
        error("No known derivative for " * function_string)
    end
end
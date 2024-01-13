using Distributions: Normal
using CUDA: CuArray

# The functions you write will accept these abstract types, allowing them
# to operate on either CPU networks or GPU networks.
abstract type DenseLayer end
abstract type DenseNetwork end

# We use Float32 because the extra precision of Float64 is almost never necessary,
# and because GPUs are better at 32-bit math.
mutable struct DenseLayerCPU <: DenseLayer
    W::Array{Float32,2}    # array of weights with size = prev_layer_nodex X this_layer_nodes
    B::Array{Float32,1}    # vector of biases with length = this_layer_nodes
    A::Array{Float32,2}    # array of stored activations with size = this_layer_nodes X batch_size
    Δ::Array{Float32,2}    # array of stored deltas with size = this_layer_nodes X batch_size
    activation::Function   # maps X → A; must operate element-wise on a matrix
    derivative::Function   # derivative of activation
end

# Identical to DenseLayerCPU, but uses CUDA arrays.
mutable struct DenseLayerGPU <: DenseLayer
    W::CuArray{Float32,2}  # array of weights with size = prev_layer_nodex X this_layer_nodes
    B::CuArray{Float32,1}  # vector of biases with length = this_layer_nodes
    A::CuArray{Float32,2}  # array of stored activations with size = this_layer_nodes X batch_size
    Δ::CuArray{Float32,2}  # array of stored deltas with size = this_layer_nodes X batch_size
    activation::Function   # maps X → A; must operate element-wise on a matrix
    derivative::Function   # derivative of activation
end

mutable struct DenseNetworkCPU <: DenseNetwork
    input_layer::Array{Float32,2}   # just stores a batch of data; doesn't need a DenseLayer object
    hidden_layers::Vector{DenseLayerCPU}
    output_layer::DenseLayerCPU
end

# Identical to DenseNetworkCPU, but uses CUDA arrays.
mutable struct DenseNetworkGPU <: DenseNetwork
    input_layer::CuArray{Float32,2} # just stores a batch of data; doesn't need a DenseLayer object
    hidden_layers::Vector{DenseLayerGPU}
    output_layer::DenseLayerGPU
end

# layer constructor
function DenseLayerCPU(layer_size::Integer, prev_layer_size::Integer; activation::Function=ReLU_activation, weight_distr=Normal(0,.1))
    W = rand(weight_distr, layer_size, prev_layer_size)
    B = zeros(layer_size)
    A = zeros(Float32, layer_size, 0) # when we actually process a batch, dimension 2 will have size = batch_size
    Δ = zeros(Float32, layer_size, 0) # when we actually process a batch, dimension 2 will have size = batch_size
    DenseLayerCPU(W, B, A, Δ, activation, get_derivative(activation))
end

# layer constructor
function DenseLayerGPU(layer_size::Integer, prev_layer_size::Integer; activation::Function=ReLU_activation, weight_distr=Normal(0,.1))
    W = rand(weight_distr, layer_size, prev_layer_size)
    B = zeros(layer_size)
    A = zeros(Float32, layer_size, 0) # when we actually process a batch, dimension 2 will have size = batch_size
    Δ = zeros(Float32, layer_size, 0) # when we actually process a batch, dimension 2 will have size = batch_size
    DenseLayerGPU(W, B, A, Δ, activation, get_derivative(activation)) # the default constructor will convert Array → CuArray
end

# network constructor
function DenseNetworkCPU(input_dim::Int64, output_dim::Int64, hidden_dims::Vector{Int64};
                hidden_activations::Union{Function,Vector{<:Function}}=ReLU_activation,
                output_activation::Function=softmax_activation, weight_distr=Normal(0,.1))
    if hidden_activations isa Function
        hidden_activations = [hidden_activations for l in 1:length(hidden_dims)]
    end
    input = zeros(Float32, input_dim, 0) # when we actually process a batch, dim 2 will have size = B
    hidden = [DenseLayerCPU(hidden_dims[1], input_dim; activation=hidden_activations[1], weight_distr=weight_distr)]
    for l in 2:length(hidden_dims)
        push!(hidden, DenseLayerCPU(hidden_dims[l], hidden_dims[l-1]; activation=hidden_activations[l], weight_distr=weight_distr))
    end
    output = DenseLayerCPU(output_dim, hidden_dims[end]; activation=output_activation, weight_distr=weight_distr)
    DenseNetworkCPU(input, hidden, output)
end

# network constructor
function DenseNetworkGPU(input_dim::Int64, output_dim::Int64, hidden_dims::Vector{Int64};
                hidden_activations::Union{Function,Vector{<:Function}}=ReLU_activation,
                output_activation::Function=softmax_activation, weight_distr=Normal(0,.1))
    if hidden_activations isa Function
        hidden_activations = [hidden_activations for l in 1:length(hidden_dims)]
    end
    input = zeros(Float32, input_dim, 0) # when we actually process a batch, dim 2 will have size = B
    hidden = [DenseLayerGPU(hidden_dims[1], input_dim; activation=hidden_activations[1], weight_distr=weight_distr)]
    for l in 2:length(hidden_dims)
        push!(hidden, DenseLayerGPU(hidden_dims[l], hidden_dims[l-1]; activation=hidden_activations[l], weight_distr=weight_distr))
    end
    output = DenseLayerGPU(output_dim, hidden_dims[end]; activation=output_activation, weight_distr=weight_distr)
    DenseNetworkGPU(input, hidden, output)
end

# Functions to transfer Networks/Layers between CPU and GPU.

function copy_to_CPU(GPU_layer::DenseLayerGPU)
    W = Array(GPU_layer.W)
    B = Array(GPU_layer.B)
    A = Array(GPU_layer.A)
    Δ = Array(GPU_layer.Δ)
    DenseLayerCPU(W, B, A, Δ, GPU_layer.activation, GPU_layer.derivative)
end

function copy_to_GPU(CPU_layer::DenseLayerCPU)
    W = Array(CPU_layer.W)
    B = Array(CPU_layer.B)
    A = Array(CPU_layer.A)
    Δ = Array(CPU_layer.Δ)
    DenseLayerGPU(W, B, A, Δ, CPU_layer.activation, CPU_layer.derivative)
end

function copy_to_CPU(GPU_network::DenseNetworkGPU)
    input = Array(GPU_network.input_layer)
    hidden = [copy_to_CPU(layer) for layer in GPU_network.hidden_layers]
    output = copy_to_CPU(GPU_network.output_layer)
    DenseNetworkCPU(input, hidden, output)
end

function copy_to_GPU(CPU_network::DenseNetworkCPU)
    input = cu(CPU_network.input_layer)
    hidden = [copy_to_GPU(layer) for layer in CPU_network.hidden_layers]
    output = copy_to_GPU(CPU_network.output_layer)
    DenseNetworkGPU(input, hidden, output)
end
using Random: shuffle


# PREDICTION & TRAINING

# Sets all activations in the network based on the input_batch using a
# single forward pass over the layers. Nothing returned.
function predict!(model::DenseNetwork, input_batch::AbstractMatrix{<:Real})

    # for the input layer
    model.input_layer = input_batch'
    
    # for hidden layers
    previous_activation_layer = model.input_layer
    for i in 1:size(model.hidden_layers, 1)  
        current_layer = model.hidden_layers[i]
        current_layer.A = current_layer.activation(current_layer.W * previous_activation_layer .+ current_layer.B)
        previous_activation_layer = current_layer.A
    end
    
    # for output layer
    model.output_layer.A = model.output_layer.activation(model.output_layer.W * previous_activation_layer .+ model.output_layer.B)
end

# Computes predictions on an entire data set, working in increments of batch_size,
# returning an array of outputs, where each row is the model's prediction on the
# corresponding row of the inputs.
# If batch_size is 0, all predictions are computed as a single batch.
function predict(model::DenseNetwork, inputs::AbstractMatrix{<:Real}; batch_size::Integer=0)
    predictions = zeros(size(inputs, 1), size(model.output_layer.A, 1))
    if batch_size == 0
        predict!(model, inputs)
        predictions = model.output_layer.A'
    else
        batch_size = (size(input,1)/length(batch_size))
        for i in 1: batch_size:N
            batch_end = min(indices[i]+batch_size-1, N)
            predict!(model, inputs[i:i + batch_size,:])
            predictions[i:batch_end,:] = model.output_layer.A'
        end
    end
  return predictions
end

# Sets all deltas in the network based on batch_targets using a single
# backward pass over the layers. Assumes that predict! was just called
# on the same batch to set the activations. Nothing returned.
function gradient!(model::DenseNetwork, batch_targets::AbstractMatrix{<:Real})
    
    #check what activation, only for output layer 
    if model.output_layer.activation == softmax_activation 
        set_CCE_output_deltas(batch_targets, model.output_layer)
    else
        set_SSE_output_deltas(batch_targets, model.output_layer)
    end
    
    # Gradients for hidden_layers
    next_deltas = model.output_layer.Δ
    next_weights = model.output_layer.W
    for i in reverse(1:size(model.hidden_layers, 1))
        current_layer = model.hidden_layers[i]
        current_layer.Δ = next_weights' * next_deltas .* current_layer.derivative(current_layer.A)
        next_deltas = current_layer.Δ
        next_weights = current_layer.W
    end
end

# Updates all weights in the network. Assumes that gradient! was just
# called to set the deltas. Nothing returned.
function update!(model::DenseNetwork, learning_rate::Real)
    
    previous_layer_A = model.input_layer
    batch_size = size(model.input_layer, 1)
    
    for i in 1:length(model.hidden_layers)
        model.hidden_layers[i].W -= ((learning_rate/batch_size) * model.hidden_layers[i].Δ * previous_layer_A')
        model.hidden_layers[i].B -= vec(((learning_rate/batch_size) * sum(model.hidden_layers[i].Δ, dims=2)))
    end 
    
    model.output_layer.W -= ((learning_rate/batch_size) * model.output_layer.Δ * previous_layer_A')
    model.output_layer.B -= vec(((learning_rate/batch_size) * sum(model.output_layer.Δ, dims=2))) 
end

# Performs mini-batch stochastic gradient descent training.
# Assumes the loss function based on the output layer's activations:
#     softmax ==> CCE; anything else ==> SSE.
# Records the loss on the entire data set at the end of each epoch
# and returns a vector of losses.
# If verbose is true, then epoch numbers and losses are printed along the way.
function train!(model::DenseNetwork, inputs::AbstractMatrix{<:Real},
                targets::AbstractMatrix{<:Real}, learning_rate::Real,
                epochs::Integer, batch_size::Integer; verbose=false)
        
    for epoch in 1:epochs
        if verbose == true
            print("epoch #" * string(epoch) * " ... ")
            flush(stdout)
        end
        N = size(inputs, 1)
        indices = shuffle(1:N)
        for  i in 1: batch_size:N
            batch_end = min(indices[i]+batch_size-1, N)
            batch_inputs = inputs[indices[i]:batch_end, :]
            batch_targets = targets[indices[i]:batch_end, :]
            predict!(model, batch_inputs)
            gradient!(model, batch_targets)
            update!(model, learning_rate)
        end
        
        outputs = predict(model, inputs)
   
        if model.output_layer.activation == softmax_activation
            loss = CCE_loss(outputs, targets) 
        else 
            loss = SSE_loss(outputs, targets)
        end

        if verbose == true                                
            println("average loss = " * string(loss))
            flush(stdout)
        end
    end
end



# Suggested helper functions; feel free to use different or additional helpers!


# Computes the activations for a hidden or output layer
# and stores them in the layer's A field.
function set_activations(curr_layer::DenseLayer, prev_layer_activations::AbstractMatrix{<:Real})
    error("unimplemented")
end

# Computes the deltas for an output layer that uses softmax
# activation and categorical cross-entropy loss, and stores
# them in the layer's Δ field.
function set_CCE_output_deltas(targets::AbstractMatrix{<:Real}, output_layer::DenseLayer)
    output_layer.Δ = output_layer.A - targets'
end

# Computes the deltas for any other type of output layer, using the
# layer's derivative-function, and stores them in the layer's Δ field.
# We'll drop the *2 since this can be absorbed into the learning rate.
function set_SSE_output_deltas(targets::AbstractMatrix{<:Real}, output_layer::DenseLayer)
    output_layer.Δ = -1 * (targets - output_layer.A) .* output_layer.derivative(output_layer.A)
end

# Computes the deltas for a hidden layer, using the,
# and stores them in the layer's Δ field.
function set_hidden_deltas(curr_layer::DenseLayer, next_layer::DenseLayer)
    error("unimplemented")
end

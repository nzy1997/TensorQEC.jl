using Test
using TensorQEC
using TensorQEC: RBM, v_to_h, h_to_v, contrastive_divergence, free_energy, train_rbm
using Random
using Statistics
using CairoMakie

# Set random seed for reproducibility
Random.seed!(10086)

# Create a simple RBM
num_visible = 4
num_hidden = 2
rbm = RBM(num_visible, num_hidden)

# Test forward and backward passes
v = rand(num_visible)
h = v_to_h(rbm, v)
v_reconstructed = h_to_v(rbm, h)

@test length(h) == num_hidden
@test length(v_reconstructed) == num_visible
@test all(0 .<= h .<= 1)
@test all(0 .<= v_reconstructed .<= 1)

# Test contrastive divergence
v_cd = contrastive_divergence(rbm, v, 1)
@test length(v_cd) == num_visible
@test all(0 .<= v_cd .<= 1)

# Test free energy
energy = free_energy(rbm, v)
@test typeof(energy) == Float64

# Test training
data = rand(10, num_visible)  # 10 samples
train_rbm(rbm, data, 0.1, 5)  # Train for 5 epochs

# Test MNIST-like data generation
function generate_mnist_like_data(n_samples::Int, image_size::Int)
    # Generate random binary images
    data = rand(n_samples, image_size) .> 0.5
    return Float64.(data)
end

# Create a larger RBM for MNIST-like data
mnist_rbm = RBM(784, 500)  # 28x28 = 784 visible nodes
mnist_data = generate_mnist_like_data(100, 784)  # 100 samples

# Train the RBM
train_rbm(mnist_rbm, mnist_data, 0.1, 10)

# Test image generation
function plot_generated_image(rbm::RBM, original::Vector{Float64})
    generated = contrastive_divergence(rbm, original, 1)
    
    # Reshape to 28x28 for display
    original_img = reshape(original, 28, 28)
    generated_img = reshape(generated, 28, 28)
    
    # Create figure with two subplots
    fig = Figure(resolution=(800, 400))
    
    # Original image
    ax1 = Axis(fig[1, 1], title="Original", aspect=DataAspect())
    heatmap!(ax1, original_img, colormap=:grays)
    hidedecorations!(ax1)
    
    # Generated image
    ax2 = Axis(fig[1, 2], title="Generated", aspect=DataAspect())
    heatmap!(ax2, generated_img, colormap=:grays)
    hidedecorations!(ax2)
    
    # Adjust layout
    colgap!(fig.layout, 20)
    
    return fig
end

# Test with a random image
test_image = rand(784)
fig = plot_generated_image(mnist_rbm, test_image)
display(fig)
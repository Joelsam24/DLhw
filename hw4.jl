# ---------------------- Q1 ----------------------
using Lux, Optimisers, Random, MLDatasets, Statistics, Plots, Flux, Zygote

# Set seed
Random.seed!(1234)

# Load FashionMNIST
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

# Preprocess
function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)

y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# Accuracy
function accuracy(model, ps, x, y)
    ŷ = model(x, ps)
    pred = Flux.onecold(ŷ)
    true_labels = Flux.onecold(y)
    mean(pred .== true_labels)

end

# Training loop
function train_model(hidden_size; epochs=10, batch_size=128, seed=1234)
    Random.seed!(seed)
    model = Lux.Chain(Lux.Dense(784, hidden_size, relu), Lux.Dense(hidden_size, 10))
    ps, st = Lux.setup(Random.default_rng(), model)
    opt = Optimisers.setup(Optimisers.Adam(0.01), ps)

    for _ in 1:epochs
        for i in 1:batch_size:size(x_train, 2)
            x = x_train[:, i:min(i+batch_size-1, end)]
            y = y_train[:, i:min(i+batch_size-1, end)]
            loss(p) = Flux.logitcrossentropy(first(model(x, p, st)), y)
            grads = Zygote.gradient(loss, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)

        end
    end
    return accuracy(model, ps, st, x_test, y_test)
end
function accuracy(model, ps, st, x, y)
    ŷ, _ = model(x, ps, st)
    mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

hidden_sizes = [10, 20, 40, 50, 100, 300]
accs = [train_model(h) for h in hidden_sizes]

plot(hidden_sizes, accs, xlabel="Hidden Size", ylabel="Accuracy", title="Q1: Hidden Size vs Accuracy", marker=:circle)

# ---------------------- Q2 ----------------------

using Lux, Optimisers, Random, MLDatasets, Statistics, Flux, Zygote, Plots

# -------------------- Load and Preprocess Data --------------------
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)
y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# -------------------- Accuracy Function --------------------
function accuracy(model, ps, st, x, y)
    ŷ, _ = model(x, ps, st)
    pred = Flux.onecold(ŷ)
    truth = Flux.onecold(y)
    mean(pred .== truth)
end

# -------------------- Training Function --------------------
function train_model_fixed(hidden_size, seed; epochs=10, batch_size=128)
    Random.seed!(seed)
    model = Lux.Chain(
        Lux.Dense(784 => hidden_size, relu),
        Lux.Dense(hidden_size => 10)
    )

    ps, st = Lux.setup(MersenneTwister(seed), model)
    opt = Optimisers.setup(Optimisers.Adam(0.001), ps)

    for epoch in 1:epochs
        for i in 1:batch_size:size(x_train, 2)
            x = x_train[:, i:min(i + batch_size - 1, end)]
            y = y_train[:, i:min(i + batch_size - 1, end)]

            loss(p) = Flux.logitcrossentropy(first(model(x, p, st)), y)
            grads = Zygote.gradient(loss, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end
    end

    return accuracy(model, ps, st, x_test, y_test)
end

# -------------------- Run Q2 --------------------
using Lux, Optimisers, Random, MLDatasets, Statistics, Flux, Zygote, Plots

# -------------------- Load and Preprocess Data --------------------
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)
y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# -------------------- Accuracy Function --------------------
function accuracy(model, ps, st, x, y)
    ŷ, _ = model(x, ps, st)
    pred = Flux.onecold(ŷ)
    truth = Flux.onecold(y)
    mean(pred .== truth)
end

# -------------------- Training Function --------------------
function train_model_fixed(hidden_size, seed; epochs=10, batch_size=128)
    Random.seed!(seed)
    model = Lux.Chain(
        Lux.Dense(784 => hidden_size, relu),
        Lux.Dense(hidden_size => 10)
    )

    ps, st = Lux.setup(MersenneTwister(seed), model)
    opt = Optimisers.setup(Optimisers.Adam(0.001), ps)

    for epoch in 1:epochs
        for i in 1:batch_size:size(x_train, 2)
            x = x_train[:, i:min(i + batch_size - 1, end)]
            y = y_train[:, i:min(i + batch_size - 1, end)]

            loss(p) = Flux.logitcrossentropy(first(model(x, p, st)), y)
            grads = Zygote.gradient(loss, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end
    end

    return accuracy(model, ps, st, x_test, y_test)
end

# -------------------- Run Q2 --------------------
hidden_size = 30
runs = 10
seeds = rand(1:10^6, runs)
accuracies = Float64[]

for (i, seed) in enumerate(seeds)
    acc = train_model_fixed(hidden_size, seed)
    push!(accuracies, acc)
    println("Run $i: Accuracy = $(round(acc * 100, digits=2))%")
end

# -------------------- Results --------------------
mean_acc = mean(accuracies)
std_acc = std(accuracies)

println("\nMean Accuracy: ", round(mean_acc * 100, digits=2), "%")
println("Standard Deviation: ", round(std_acc * 100, digits=2), "%")

scatter(
    1:runs, accuracies,
    xlabel = "Run Index",
    ylabel = "Test Accuracy",
    title = "Q2: Accuracy vs Random Initialization (Hidden Size = 30)",
    marker = :circle,
    legend = false
)
hline!([mean_acc], linestyle = :dash, label = "Mean Accuracy")


# ---------------------- Q4 ----------------------

using Lux, Optimisers, Random, MLDatasets, Statistics, Flux, Zygote

Random.seed!(1234)

train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)

y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

function accuracy(model, ps, st, x, y)
    ŷ, _ = model(x, ps, st)
    pred = Flux.onecold(ŷ)
    truth = Flux.onecold(y)
    mean(pred .== truth)
end

function get_batches(x, y, batch_size)
    N = size(x, 2)
    idx = shuffle(1:N)
    [((x[:, idx[i:min(i+batch_size-1, N)]], y[:, idx[i:min(i+batch_size-1, N)]])) for i in 1:batch_size:N]
end

function train_model(batch_size, decay_rate)
    model = Lux.Chain(
        Lux.Dense(784 => 50, relu),
        Lux.Dense(50 => 10)
    )

    ps, st = Lux.setup(MersenneTwister(1234), model)

    for epoch in 1:10
        η = 0.01 * decay_rate^(epoch - 1)
        opt = Optimisers.setup(Optimisers.Adam(η), ps)  # ✅ correct

        for i in 1:batch_size:size(x_train, 2)
            x = x_train[:, i:min(i + batch_size - 1, end)]
            y = y_train[:, i:min(i + batch_size - 1, end)]

            loss(p) = Flux.logitcrossentropy(first(model(x, p, st)), y)
            grads = Zygote.gradient(loss, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end
    end

    return accuracy(model, ps, st, x_test, y_test)
end


batch_sizes = [16, 32, 64]
decay_rates = [0.9, 0.95, 0.99]
results = Dict()

for b in batch_sizes
    for d in decay_rates
        acc = train_model(b, d)
        results[(b, d)] = acc
        println("Batch $b | Decay $d -> Accuracy: $(round(acc * 100, digits=2))%")
    end
end

# Sort the results dictionary by accuracy (value) in descending order
sorted_results = sort(collect(pairs(results)); by = x -> x.second, rev = true)

# Extract the best performing (batch_size, decay_rate) configuration
best_entry = first(sorted_results)
best_config = best_entry.first      # (batch_size, decay_rate)
best_accuracy = best_entry.second   # final accuracy as Float64

# Display the result
println("Best configuration:")
println("  Batch Size = $(best_config[1])")
println("  Decay Rate = $(best_config[2])")
println("  Accuracy   = $(round(best_accuracy * 100, digits=2))%")



# ---------------------- Q5 ----------------------

using Lux, Optimisers, Random, MLDatasets, Statistics, Flux, Zygote

Random.seed!(1234)

train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)

y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

function accuracy(model, ps, st, x, y)
    ŷ, _ = model(x, ps, st)
    pred = Flux.onecold(ŷ)
    truth = Flux.onecold(y)
    mean(pred .== truth)
end

function get_batches(x, y, batch_size)
    N = size(x, 2)
    idx = shuffle(1:N)
    [((x[:, idx[i:min(i+batch_size-1, N)]], y[:, idx[i:min(i+batch_size-1, N)]])) for i in 1:batch_size:N]
end

# Based on Q4 best results
batch_size = 32 
decay_rate = 0.95
initial_lr = 0.01
epochs = 25

model = Chain(Dense(28^2 => 50, relu), Dense(50 => 10))
ps, st = Lux.setup(Random.default_rng(), model)

for epoch in 1:epochs
    lr = initial_lr * decay_rate^(epoch - 1)
    opt = Optimisers.setup(Optimisers.Descent(lr))
    for (x_batch, y_batch) in get_batches(x_train, y_train, batch_size)
        grads = gradient(ps) do
            ŷ = model(x_batch, ps)
            Flux.logitcrossentropy(ŷ, y_batch)
        end
        ps, st = Optimisers.update(opt, ps, grads, st)
    end
    acc = accuracy(model, ps, x_test, y_test)
    println("Epoch $epoch: Test Accuracy = $(round(acc * 100, digits=2))% | LR = $(round(lr, digits=5))")
end




# Set seed
Random.seed!(1234)

# Load FashionMNIST
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

# Preprocess
function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)

y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# Accuracy
function accuracy(model, ps, x, y)
    ŷ = model(x, ps)
    pred = Flux.onecold(ŷ)
    true = Flux.onecold(y)
    mean(pred .== true)
end

# Training loop
function train!(model, ps, st, opt, x, y, epochs)
    for epoch in 1:epochs
        grads = gradient(ps) do
            ŷ = model(x, ps)
            Flux.logitcrossentropy(ŷ, y)
        end
        ps, st = Optimisers.update(opt, ps, grads, st)
    end
    return ps, st
end

# Hidden sizes to try
hidden_sizes = [10, 20, 40, 50, 100, 300]
accuracies = Float64[]

for h in hidden_sizes
    model = Chain(Dense(28^2 => h, relu), Dense(h => 10))
    ps, st = Lux.setup(Random.default_rng(), model)
    opt = Optimisers.Adam()

    ps, st = train!(model, ps, st, opt, x_train, y_train, 10)
    acc = accuracy(model, ps, x_test, y_test)
    push!(accuracies, acc)
    println("Hidden size: $h -> Accuracy: $(round(acc*100, digits=2))%")
end

# Plot
plot(hidden_sizes, accuracies, marker=true, xlabel="Hidden Layer Size", ylabel="Test Accuracy", title="Accuracy vs Hidden Layer Size")


# ---------------------- Q2 ----------------------

using Lux, Optimisers, Random, MLDatasets, Statistics, Plots, Flux, Zygote

# Seed
Random.seed!(1234)

# Load FashionMNIST
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

# Preprocess
function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)

y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# Accuracy
function accuracy(model, ps, x, y)
    ŷ = model(x, ps)
    pred = Flux.onecold(ŷ)
    true = Flux.onecold(y)
    mean(pred .== true)
end

# Training
function train!(model, ps, st, opt, x, y, epochs)
    for epoch in 1:epochs
        grads = gradient(ps) do
            ŷ = model(x, ps)
            Flux.logitcrossentropy(ŷ, y)
        end
        ps, st = Optimisers.update(opt, ps, grads, st)
    end
    return ps, st
end

# Fixed hidden size
hidden_size = 30
runs = 10
accuracies = Float64[]

for run in 1:runs
    model = Chain(Dense(28^2 => hidden_size, relu), Dense(hidden_size => 10))
    ps, st = Lux.setup(MersenneTwister(run), model)  # Different init
    opt = Optimisers.Adam()

    ps, st = train!(model, ps, st, opt, x_train, y_train, 10)
    acc = accuracy(model, ps, x_test, y_test)
    push!(accuracies, acc)
    println("Run $run: Accuracy = $(round(acc*100, digits=2))%")
end

mean_acc = mean(accuracies)
std_acc = std(accuracies)

println("Mean Accuracy: $(round(mean_acc*100, digits=2))%")
println("Std Deviation: $(round(std_acc*100, digits=2))")

# Plot accuracies
scatter(1:runs, accuracies, xlabel="Run", ylabel="Test Accuracy", title="Accuracy Fluctuations (Hidden Size 30)")


# ---------------------- Q3 ----------------------

using Lux, Optimisers, Random, MLDatasets, Statistics, Flux, Zygote, Plots

Random.seed!(1234)

train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)

y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

function accuracy(model, ps, x, y)
    ŷ = model(x, ps)
    pred = Flux.onecold(ŷ)
    true = Flux.onecold(y)
    mean(pred .== true)
end

# Mini-batch iterator
function get_batches(x, y, batch_size)
    N = size(x, 2)
    idx = shuffle(1:N)
    [((x[:, idx[i:min(i+batch_size-1, N)]], y[:, idx[i:min(i+batch_size-1, N)]])) for i in 1:batch_size:N]
end

# Model
model = Chain(Dense(28^2 => 50, relu), Dense(50 => 10))
ps, st = Lux.setup(Random.default_rng(), model)

# Training with learning rate decay
epochs = 25
batch_size = 32
initial_lr = 0.01
decay_rate = 0.95
opt = Optimisers.setup(Optimisers.Descent(initial_lr))

for epoch in 1:epochs
    batches = get_batches(x_train, y_train, batch_size)
    lr = initial_lr * decay_rate^(epoch - 1)
    opt = Optimisers.setup(Optimisers.Descent(lr))

    for (x_batch, y_batch) in batches
        grads = gradient(ps) do
            ŷ = model(x_batch, ps)
            Flux.logitcrossentropy(ŷ, y_batch)
        end
        ps, st = Optimisers.update(opt, ps, grads, st)
    end

    acc = accuracy(model, ps, x_test, y_test)
    println("Epoch $epoch: Test Accuracy = $(round(acc * 100, digits=2))% | LR = $(round(lr, digits=5))")
end


# ---------------------- Q4 ----------------------

using Lux, Optimisers, Random, MLDatasets, Statistics, Flux, Zygote

Random.seed!(1234)

train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)

y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

function accuracy(model, ps, x, y)
    ŷ = model(x, ps)
    pred = Flux.onecold(ŷ)
    true = Flux.onecold(y)
    mean(pred .== true)
end

function get_batches(x, y, batch_size)
    N = size(x, 2)
    idx = shuffle(1:N)
    [((x[:, idx[i:min(i+batch_size-1, N)]], y[:, idx[i:min(i+batch_size-1, N)]])) for i in 1:batch_size:N]
end

function train_model(batch_size, decay_rate)
    model = Chain(Dense(28^2 => 50, relu), Dense(50 => 10))
    ps, st = Lux.setup(Random.default_rng(), model)
    initial_lr = 0.01
    epochs = 10

    for epoch in 1:epochs
        batches = get_batches(x_train, y_train, batch_size)
        lr = initial_lr * decay_rate^(epoch - 1)
        opt = Optimisers.setup(Optimisers.Descent(lr))

        for (x_batch, y_batch) in batches
            grads = gradient(ps) do
                ŷ = model(x_batch, ps)
                Flux.logitcrossentropy(ŷ, y_batch)
            end
            ps, st = Optimisers.update(opt, ps, grads, st)
        end
    end

    return accuracy(model, ps, x_test, y_test)
end

batch_sizes = [16, 32, 64]
decay_rates = [0.9, 0.95, 0.99]
results = Dict()

for b in batch_sizes
    for d in decay_rates
        acc = train_model(b, d)
        results[(b, d)] = acc
        println("Batch $b | Decay $d -> Accuracy: $(round(acc * 100, digits=2))%")
    end
end

# Find best combo
best = findmax(values(results))
best_key = filter(k -> results[k] == best[1], keys(results))[1]
println("Best combo: Batch Size = $(best_key[1]), Decay Rate = $(best_key[2]) with Accuracy = $(round(best[1]*100, digits=2))%")


# ---------------------- Q5 ----------------------

using Lux, Optimisers, Random, MLDatasets, Statistics, Flux, Zygote

# -------------------- Load and preprocess data --------------------
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
x_test = preprocess(test_x)
y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# -------------------- Accuracy function --------------------
function accuracy(model, ps, st, x, y)
    ŷ, _ = model(x, ps, st)
    pred = Flux.onecold(ŷ)
    truth = Flux.onecold(y)
    mean(pred .== truth)
end

# -------------------- Training function with decay --------------------
function train_with_best_config(hidden_size, batch_size, initial_lr, decay_rate;
                                seed = 1234, epochs = 25)
    Random.seed!(seed)
    model = Lux.Chain(
        Lux.Dense(28^2 => hidden_size, relu),
        Lux.Dense(hidden_size => 10)
    )

    ps, st = Lux.setup(MersenneTwister(seed), model)

    for epoch in 1:epochs
        lr = initial_lr * decay_rate^(epoch - 1)
        opt = Optimisers.setup(Optimisers.Adam(lr), ps)

        for i in 1:batch_size:size(x_train, 2)
            x = x_train[:, i:min(i + batch_size - 1, end)]
            y = y_train[:, i:min(i + batch_size - 1, end)]

            loss(p) = Flux.logitcrossentropy(first(model(x, p, st)), y)
            grads = Zygote.gradient(loss, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end

        println("Epoch $epoch done. Learning rate: $(round(lr, digits=5))")
    end

    acc = accuracy(model, ps, st, x_test, y_test)
    return acc
end

# -------------------- Use best hyperparameters --------------------
# Replace these with actual best values from your Q4
best_batch_size = 64
best_decay = 0.99
initial_lr = 0.01
hidden_size = 50

# -------------------- Train and evaluate --------------------
final_acc = train_with_best_config(hidden_size, best_batch_size, initial_lr, best_decay)
println("\nFinal Test Accuracy using best config: $(round(final_acc * 100, digits=2))%")

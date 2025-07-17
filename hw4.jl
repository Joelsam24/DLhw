using Lux, Optimisers, Random, MLDatasets, Statistics, Plots, Flux, Zygote

# -------------------------Question 1 ------------------------------
Random.seed!(1234)
train_x, train_y = FashionMNIST.traindata()
ytest_x, test_y = FashionMNIST.testdata()

function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
y_train = Flux.onehotbatch(train_y .+ 1, 1:10)

x_test = preprocess(test_x)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

learning_rate = 0.01
batch_size = 128
epochs = 10

function accuracy(model, x, y)
    ŷ = model(x)
    mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

function train_model(hidden_size; seed=1234)
    Random.seed!(seed)
    model = Lux.Chain(Lux.Dense(28^2, hidden_size, relu), Lux.Dense(hidden_size, 10))
    ps, st = Lux.setup(Random.default_rng(), model)
    opt = Optimisers.setup(Optimisers.Adam(learning_rate), ps)

    for epoch in 1:epochs
        for i in 1:batch_size:size(x_train, 2)
            last = min(i + batch_size - 1, size(x_train, 2))
            x_batch = x_train[:, i:last]
            y_batch = y_train[:, i:last]

            function loss_fun(p)
                ŷ, _ = model(x_batch, p, st)
                Flux.logitcrossentropy(ŷ, y_batch)
            end

            grads = Zygote.gradient(loss_fun, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end
    end

    ŷ_test, _ = model(x_test, ps, st)
    return accuracy((x) -> first(model(x, ps, st)), x_test, y_test)
end

hidden_sizes = [10, 20, 40, 50, 100, 300]
accuracies = [train_model(h) for h in hidden_sizes]

plot(hidden_sizes, accuracies, xlabel="Hidden Layer Size", ylabel="Test Accuracy", title="Test Accuracy vs Hidden Layer Size", lw=2, marker=:circle)

# -------------------------Question 2 ------------------------------
hidden_size_fixed = 30
n_runs = 10
seeds = rand(1:10^6, n_runs)
accuracies_random_init = [train_model(hidden_size_fixed, seed=s) for s in seeds]

mean_accuracy = mean(accuracies_random_init)
std_accuracy = std(accuracies_random_init)

println("Mean Accuracy: ", round(mean_accuracy * 100, digits=2), "%")
println("Standard Deviation: ", round(std_accuracy * 100, digits=2), "%")

scatter(1:n_runs, accuracies_random_init, xlabel = "Run Index", ylabel = "Test Accuracy", title = "Test Accuracy for 10 Random Initializations (Hidden Size = 30)", legend = false, marker = :circle)
hline!([mean_accuracy], label="Mean", linestyle=:dash)

# -------------------------Question 3 ------------------------------
batch_size_q3 = 32
epochs_q3 = 25

function learning_rate_schedule(initial_lr, epoch, decay_rate=0.9)
    return initial_lr * (decay_rate ^ (epoch - 1))
end

function train_with_decay(hidden_size; seed=1234, initial_lr=0.01, decay_rate=0.9)
    Random.seed!(seed)
    model = Lux.Chain(Lux.Dense(28^2, hidden_size, relu), Lux.Dense(hidden_size, 10))
    ps, st = Lux.setup(Random.default_rng(), model)

    for epoch in 1:epochs_q3
        current_lr = learning_rate_schedule(initial_lr, epoch, decay_rate)
        opt = Optimisers.setup(Optimisers.Adam(current_lr), ps)

        for i in 1:batch_size_q3:size(x_train, 2)
            last = min(i + batch_size_q3 - 1, size(x_train, 2))
            x_batch = x_train[:, i:last]
            y_batch = y_train[:, i:last]

            function loss_fun(p)
                ŷ, _ = model(x_batch, p, st)
                Flux.logitcrossentropy(ŷ, y_batch)
            end

            grads = Zygote.gradient(loss_fun, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end

        println("Epoch $epoch complete. Learning rate: $(round(current_lr, digits=5))")
    end

    ŷ_test, _ = model(x_test, ps, st)
    return accuracy((x) -> first(model(x, ps, st)), x_test, y_test)
end

final_accuracy = train_with_decay(50)
println("Final Test Accuracy (Hidden Size = 50): ", round(final_accuracy * 100, digits=2), "%")

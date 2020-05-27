using ComputationalResources
using MLJ
using DelimitedFiles
@load PerceptronClassifier pkg = ScikitLearn

using MLDatasets

train_X, tr_y = MNIST.traindata(Float64)
test_X, ts_y = MNIST.testdata(Float64)

tr_X = Array{Float64,2}(
    undef,
    size(train_X, 3),
    size(train_X, 1) * size(train_X, 2),
)
for i = 1:size(train_X, 3)
    tr_X[i, :] = collect(Iterators.flatten(train_X[:, :, i]))
end

ts_X =
    Array{Float64,2}(undef, size(test_X, 3), size(test_X, 1) * size(test_X, 2))
for i = 1:size(test_X, 3)
    ts_X[i, :] = collect(Iterators.flatten(test_X[:, :, i]))
end

#lm = tr_X
# open("ml/data/opdig_lrg_tr.csv", "w") do io
#     writedlm(io, lm[1:100, :], ',')
# end

function get_block_indices_long(cornerindices::Array{Int64,2}, dims::Array{Int64,1})
    block_row_start::Int64 = cornerindices[1, 1]
    block_col_start::Int64 = cornerindices[1, 2]
    block_row_end::Int64 = cornerindices[2, 1]
    block_col_end::Int64 = cornerindices[2, 2]

    tl_corner::Int64 = dims[1] * (block_row_start - 1) + block_col_start

    width::Int64 = block_col_end - block_col_start + 1
    height::Int64 = block_row_end - block_row_start + 1

    r_seq_1::Array{Int64,1} = collect(tl_corner:1:tl_corner+width-1)
    r_seq = deepcopy(r_seq_1)

    h_iter::Int64 = height - 1
    for i in 1:h_iter
        append!(r_seq, r_seq_1 .+ (dims[1] * i))
    end

    return (r_seq)
end

function long_matrix_subaverage(X::Array{Float64,2}, dims::Array{Int64,1}, sddims::Array{Int64,1} = [2, 2])
    num_matrices::Int64 = size(X, 1)
    if dims[1] * dims[2] != size(X, 2)
        throw(DimensionMismatch("Desired number of elements in matrix is not equal to number in array"))
    end
    num_sd_axes::Array{Int64,1} = floor.(Int64, dims ./ sddims)
    subav::Array{Float64,2} = Array{Float64,2}(undef, num_matrices, prod(num_sd_axes))
    for matrix_number = 1:num_matrices
        arr_subind::Int64 = 1
        mat_long::Array{Float64,1} = X[matrix_number, :]
        for row_sub in 1:num_sd_axes[2]
            for col_sub in 1:num_sd_axes[1]
                brs::Int64 = 1 + (row_sub - 1) * sddims[1]
                bre::Int64 = (row_sub) * sddims[1]
                crs::Int64 = 1 + (col_sub - 1) * sddims[2]
                cre::Int64 = (col_sub) * sddims[2]
                indices::Array{Int64,2} = [[brs, bre] [crs, cre]]
                submat_long::Array{Float64,1} = mat_long[get_block_indices_long(indices, dims)]
                submat_avg::Float64 = mean(submat_long)
                if submat_avg < eps()
                    submat_avg = 0.0
                end
                subav[matrix_number, arr_subind] = submat_avg
                arr_subind += 1
            end
        end
    end
    return(subav)
end

# tr = readdlm("ml/data/optdigitstra.csv", ',')
# tr_X = tr[:, 1:64]
# tr_y = tr[:, 65]
#
# ts = readdlm("ml/data/optdigitstes.csv", ',')
# ts_X = ts[:, 1:64]
# ts_y = ts[:, 65]

tr_sz = 50000
ts_sz = 5000

tr_X_11 = long_matrix_subaverage(tr_X[1:tr_sz,:], [28, 28], [1,1])
ts_X_11 = long_matrix_subaverage(ts_X[1:ts_sz,:], [28, 28], [1,1])
tr_X_22 = long_matrix_subaverage(tr_X[1:tr_sz,:], [28, 28], [2,2])
ts_X_22 = long_matrix_subaverage(ts_X[1:ts_sz,:], [28, 28], [2,2])
tr_X_33 = long_matrix_subaverage(tr_X[1:tr_sz,:], [28, 28], [3,3])
ts_X_33 = long_matrix_subaverage(ts_X[1:ts_sz,:], [28, 28], [3,3])

#tr_X = hcat(tr_X_22, tr_X_33)
#ts_X = hcat(ts_X_22, ts_X_33)

tr_X = tr_X[1:tr_sz,:]
ts_X = ts_X[1:ts_sz,:]

tr_y = tr_y[1:tr_sz]
ts_y = ts_y[1:ts_sz]

model = PerceptronClassifier(max_iter = 10_000, early_stopping = true, tol = 1e-6)


num_features = size(tr_X, 2)

r1 = range(model, :eta0, lower = 1.0, upper = 1.5)
st_boost = TunedModel(model = model, range = r1, tuning = Grid(resolution = 5), acceleration=CPU1(), acceleration_resampling=CPU1())

train_pred = table(tr_X)
train_out = categorical(tr_y)

test_pred = table(ts_X)
test_out = categorical(ts_y)
#mach = machine(model, train_pred, train_out)
mach = machine(st_boost, train_pred, train_out)

fit!(mach)
fi_p = fitted_params(mach).best_model

yp = predict(mach, test_pred)
mcr = misclassification_rate(yp, test_out)
conf = confusion_matrix(yp, test_out)

pcm = conf[:,:]
dvs = sum(pcm, dims=1)
props = pcm ./ dvs

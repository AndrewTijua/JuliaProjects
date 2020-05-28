using LinearAlgebra
using Images
using ImageTransformations
using CoordinateTransformations

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
    for i = 1:h_iter
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
        for row_sub = 1:num_sd_axes[2]
            for col_sub = 1:num_sd_axes[1]
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
    return (subav)
end

function long_matrix_to_matrix(X::Array{Float64,2}, dims::Array{Int64,1})
    num_matrices::Int64 = size(X, 1)
    if dims[1] * dims[2] != size(X, 2)
        throw(DimensionMismatch("Desired number of elements in matrix is not equal to number in array"))
    end
    matrix_list::Array{Float64,3} = Array{Float64,3}(undef, dims[1], dims[2], num_matrices)
    for matrix_number = 1:num_matrices
        matrix_list[:, :, matrix_number] = permutedims(reshape(X[matrix_number, :], (dims[1], dims[2])))
    end
    return (matrix_list)
end

function augment_data_rotate(X::Array{Float64,3}, thetarange = collect(0.01*π:0.02*π:π*0.25), initial_matrices::Int64 = size(X, 3))
    num_augments_per_matrix::Int64 = size(thetarange, 1)
    index_aug::Array{Int64,1} = Array{Int64,1}(undef, num_augments_per_matrix * initial_matrices)
    for matrix_number = 1:initial_matrices
        for rotation_ind in 1:size(thetarange, 1)
            X = cat(X, replace(imrotate(X[:, :, matrix_number], thetarange[rotation_ind], axes(X[:, :, matrix_number])), NaN => 0.0); dims = 3)
            index_aug[num_augments_per_matrix * (matrix_number-1)+rotation_ind] = matrix_number
        end
    end
    return (X, index_aug)
end

function augment_data_shift(X::Array{Float64,3}, shiftrange = [-1, 0, 1], initial_matrices::Int64 = size(X, 3))
    num_augments_per_matrix::Int64 = 1
    for matrix_number = 1:initial_matrices
        for shift_1 in shiftrange
            for shift_2 in shiftrange
                trs = Translation(shift_1, shift_2)
                if (abs(shift_1) + abs(shift_2) > 0)
                    X = cat(X, warp(X[:, :, matrix_number], trs, indices_spatial(X[:, :, matrix_number]), 0); dims = 3)
                end
            end
        end
    end
    return (X)
end

function matrix_array_to_long_list(m_array::Array{Float64,3})
    X = Array{Float64,2}(undef, size(m_array, 3), size(m_array, 1) * size(m_array, 2))
    for i = 1:size(m_array, 3)
        X[i, :] = collect(Iterators.flatten(m_array[:, :, i]))
    end
    return (X)
end

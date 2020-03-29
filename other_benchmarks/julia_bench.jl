using BenchmarkTools
using Random
using LinearAlgebra


BenchmarkTools.DEFAULT_PARAMETERS.evals = 20

# define arrays and matrices
const rows, cols = 500, 600
const reduceRows, reduceCols = Int(rows / 5), Int(cols / 6)

const float_arrayA = rand(rows * cols)
const float_arrayB = rand(rows * cols)
const small_int_matrixA = rand(1:10, reduceRows, reduceCols)
const small_int_matrixB = rand(1:10, reduceRows, reduceCols)
const float_matrixA = rand(rows, cols)
const float_matrixC = rand(cols, rows)
const small_float_matrixA = rand(reduceRows, reduceCols)
const small_float_matrixB = rand(reduceRows, reduceCols)

test1_name = "Element-wise sum of two $(reduceRows)x$(reduceCols) matrices (int), (1000 loops)"
function test1()
    for _ in 1:1000
        _ = small_int_matrixA + small_int_matrixB
    end
end

test2_name = "Element-wise multiplication of two $(reduceRows)x$(reduceCols) matrices (float64), (1000 loops)"
function test2()
    for _ in 1:1000
        _ = small_float_matrixA .* small_float_matrixB
    end
end

test3_name = "Dot (scalar) product of two $(rows * cols) arrays (float64), (1000 loops)"
function test3()
    for _ in 1:1000
        _ = dot(float_arrayA, float_arrayB)
    end
end

test4_name = "Matrix product of $(rows)x$(cols) and $(cols)x$(rows) matrices (float64)"
function test4()
    _ = float_matrixA * float_matrixC
end

test5_name = "L2 norm of $(rows)x$(cols) matrix (float64), (1000 loops)"
function test5()
    for _ in 1:1000
        _ = norm(float_matrixA)
    end
end

test6_name = "Sort of $(rows)x$(cols) matrix (float64)"
function test6()
    _ = sort(float_matrixA, dims = 2)
end


println(test1_name)
@btime test1()

println(test2_name)
@btime test2()

println(test3_name)
@btime test3()

println(test4_name)
@btime test4()

println(test5_name)
@btime test5()

println(test6_name)
@btime test6()

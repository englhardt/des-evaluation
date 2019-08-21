JULIA_ENV = joinpath(@__DIR__, "..", "..")
data_root = joinpath(@__DIR__, "..", "..", "data")
data_input_root = joinpath(data_root, "input", "processed")
data_output_root = joinpath(data_root, "output")

worker_list = [("localhost", 1)]
exeflags = `--project="$JULIA_ENV"`
sshflags= `-i path/to/remote/ssh/key`

fmt_string = "[{name} | {date} | {level}]: {msg}"
loglevel = "debug"

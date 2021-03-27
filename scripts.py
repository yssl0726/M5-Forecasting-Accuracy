import os

# 每一条system函数执行会创建不同的子进程
# 由于脚本执行有顺序性，所以要保证多条命令在同一个子进程中运行，仅用一个system()
os.system("cd ./codes && python ./fe.py && python ./train_state.py && python ./train_store.py && python ./fusion.py")



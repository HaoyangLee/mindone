import os
import socket
import sys

"""
On modelarts, usage example:
python /home/ma-user/modelarts/user-job-dir/mindone/mindone/utils/modelarts/msrun.py [WORK_DIR] [SCRIPT_NAME]
python /home/ma-user/modelarts/user-job-dir/mindone/mindone/utils/modelarts/msrun.py mindone/examples/opensora_cai train_t2v.py
"""


def query_host_ip(host_addr):
    try:
        ip = socket.gethostbyname(host_addr)
    finally:
        return ip


def run():
    ip_addr = os.getenv("VC_WORKER_HOSTS").split(",")
    print("host names:", ip_addr)
    print(os.environ)
    ip_addr_list = []
    for i in ip_addr:
        host_addr_ip = query_host_ip(i)
        ip_addr_list.append(host_addr_ip)
    print("ip address list:", ip_addr_list)
    master_addr = ip_addr_list[0]
    node_rank = int(os.getenv("VC_TASK_INDEX"))
    print(f"=======> {sys.argv}", flush=True)
    work_dir = sys.argv[1]  # e.g. mindone/examples/opensora_cai
    script_name = sys.argv[2]  # e.g. train_t2v.py
    args = " ".join(sys.argv[3:])
    print("job start with ")

    # install packages before launching training on modelarts
    # return_code = os.system(
    #     f"bash /home/ma-user/modelarts/user-job-dir/{work_dir}/ma-pre-start.sh")

    command = f"bash /home/ma-user/modelarts/user-job-dir/mindone/mindone/utils/modelarts/run_train_modelarts.sh \
        {master_addr} {node_rank} {work_dir} {script_name} {args}"
    print("Running command:", command)
    os.system(command)


if __name__ == "__main__":
    run()

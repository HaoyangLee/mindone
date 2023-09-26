import sys

from common import RunAscendLog
from common import RankTableEnv

from rank_table import RankTable, RankTableTemplate2

from manager import FMKManager

if __name__ == '__main__':
    log = RunAscendLog.setup_run_ascend_logger()

    if len(sys.argv) <= 1:
        log.error('there are not enough args')
        sys.exit(1)

    train_command = sys.argv[1:]
    log.info('training command')
    log.info(train_command)

    rank_table_path_origin = RankTableEnv.get_rank_table_template2_file_path()
    RankTable.wait_for_available(rank_table_path_origin)

    rank_table = RankTableTemplate2(rank_table_path_origin)
    RankTableEnv.set_rank_table_env(rank_table.get_rank_table_path())

    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    current_instance = RankTable.convert_server_to_instance(server)

    fmk_manager = FMKManager(current_instance)
    fmk_manager.run(rank_table.get_device_num(), train_command)
    return_code = fmk_manager.monitor()

    fmk_manager.destroy()

    sys.exit(return_code)

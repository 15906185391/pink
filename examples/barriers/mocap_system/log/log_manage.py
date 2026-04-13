from loguru import logger
import os
from datetime import datetime

logger.remove(handler_id=None)  # 清除之前的设置

home_teleop_dir = os.path.join(os.path.expanduser("~"), ".teleop")
log_dir = os.path.join(home_teleop_dir, "log", "log_data")
os.makedirs(log_dir, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")


class logUnit():
    def __init__(self,name):
        self.name = name
        pass
    
    def update_log_info(self, log_type, info):
        info = self.name + " : " + info
        if(log_type=="WARNING"):
            logger.warning(info)
        elif(log_type=="INFO"):
            logger.info(info)
        elif(log_type=="ERROR"):
            logger.error(info)
        elif(log_type=="EXCEPTION"):
            logger.exception(info)


class logManage():
    def __init__(self):
        logger.add(os.path.join(log_dir, f'info_{ts}.log'), level='INFO', mode='w',
                   format='{time} {level} {message}', rotation='00:00', retention='1 days',
                   compression='zip', encoding='utf-8', enqueue=True)
        logger.add(os.path.join(log_dir, f'error_{ts}.log'), level='ERROR',
                   format='{time} {level} {message}', rotation='00:00', retention='1 days',
                   compression='zip', encoding='utf-8', enqueue=True)
        logger.add(os.path.join(log_dir, f'warning_{ts}.log'), level='WARNING',
                   format='{time} {level} {message}', rotation='00:00', retention='1 days',
                   compression='zip', encoding='utf-8', enqueue=True)

        self.tracker_state = logUnit("TRACKER STATE")
        self.mocap_info = logUnit("MOCAP INFO")
        self.target_pos_eul = logUnit("TARGET POS EUL INFO")
        self.real_pos_eul = logUnit("REAL POS EUL INFO")
        self.end_pos_error = logUnit("REAL POS EUL INFO")
        self.qpos_info = logUnit("REAL POS EUL INFO")
        self.import_error = logUnit("IMPORT ERROR INFO")
        self.mocap_state = logUnit("MOCAP STATE INFO")


log_m = logManage()


if __name__ =="__main__":
    s = logManage()

    s.tracker_state.update_log_info("INFO","11111")

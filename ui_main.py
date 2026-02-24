import sys
import logging
from datetime import datetime
from PyQt6 import QtWidgets
from ui_main_window import MainWindow

log_filename = f"crowd_analysis_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('CrowdAnalysis')
logger.info(f"Logging initialized. Log file: {log_filename}")

if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("APPLICATION STARTING")
        logger.info("="*60)
        
        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow(log_filename)
        win.show()
        
        logger.info("GUI shown successfully")
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical("Fatal error starting application", exc_info=True)
        print(f"\n\nFATAL ERROR: {e}")
        print(f"Check log file: {log_filename}")
        raise
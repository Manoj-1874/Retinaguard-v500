import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class LoggerWriter:
    def __init__(self, logger_func):
        self.logger_func = logger_func
        self.buf = []
        
    def write(self, msg):
        if msg == '\n':
            if self.buf:
                self.logger_func("".join(self.buf))
                self.buf = []
            else:
                self.logger_func("")
        else:
            self.buf.append(msg)
            
    def flush(self):
        if self.buf:
            self.logger_func("".join(self.buf))
            self.buf = []

sys.stdout = LoggerWriter(logger.info)

print("Test line 1")
print("Test line 2", flush=True)
print("Test line 3", end=" -> ", flush=True)
print("PASS", flush=True)

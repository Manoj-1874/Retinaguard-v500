import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# I had replaced `print(` with `logger.info(`. But there were originally some actual `logger.info(` calls!
# It's fine if they all become `log_print(`.
code = re.sub(r'logger\.info\(', r'log_print(', code)

# Prepend the log_print definition
prefix = """import logging

def log_print(*args, **kwargs):
    logger = logging.getLogger(__name__)
    msg = " ".join(str(a) for a in args)
    logger.info(msg)

"""

if "def log_print(" not in code:
    code = prefix + code

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

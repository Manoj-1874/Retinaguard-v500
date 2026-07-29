import logging
from flask import Flask

# This is what app.py does:
logging.getLogger('werkzeug').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.info("THIS IS A TEST LOG")
